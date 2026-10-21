#if __has_include(<SDL2/SDL.h>)
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>

// Размеры виртуального холста и геометрия шара
constexpr int W = 520;
constexpr int H = 560;
constexpr int GLOBE_CX = W / 2;
constexpr int GLOBE_CY = H / 2 - 25;
constexpr int RADIUS = 215;

// 12 288 снежинок (кратно 4 для 128-битных регистров ARM NEON v0.4s)
constexpr int NUM_PARTICLES = 12288;

// Параметры пространственной сетки (ячейка 4x4 пикселя)
constexpr int CELL_SHIFT = 2;
constexpr int GRID_W = (W >> CELL_SHIFT) + 2;
constexpr int GRID_H = (H >> CELL_SHIFT) + 2;
constexpr int NUM_CELLS = GRID_W * GRID_H;

// Выровненные по 16 байт массивы SoA (Structure of Arrays)
alignas(16) float px[NUM_PARTICLES];
alignas(16) float py[NUM_PARTICLES];
alignas(16) float vx[NUM_PARTICLES];
alignas(16) float vy[NUM_PARTICLES];
alignas(16) float active[NUM_PARTICLES]; // 1.0f = парит в воде, 0.0f = лежит в сугробе
alignas(16) float mass[NUM_PARTICLES];   // Индивидуальная плавучесть (0.45 .. 1.35)

// 0 = вода, 1 = сугроб, 2 = ёлка, 3 = стекло шара
uint8_t snowGrid[H][W];
uint32_t pixels[W * H];

// Пространственная хэш-сетка для упругих столкновений в воздухе
int cellHead[NUM_CELLS];
int nextParticle[NUM_PARTICLES];

// Быстрый генератор случайных чисел Xorshift32
uint32_t rngState = 0x9E3779B9;
inline uint32_t FastRand() {
    uint32_t x = rngState;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rngState = x;
}
inline float RandFloat(float minV, float maxV) {
    return minV + (FastRand() & 0xFFFF) / 65535.0f * (maxV - minV);
}

// ============================================================================
// АССЕМБЛЕРНОЕ ЯДРО ARM64 NEON (AArch64) — по 4 снежинки за такт
// ============================================================================
void StepPhysicsASM(float drag, float ax, float ay) {
#if defined(__aarch64__)
    size_t loops = NUM_PARTICLES / 4;
    float* pPx = px;
    float* pPy = py;
    float* pVx = vx;
    float* pVy = vy;
    const float* pAct = active;
    const float* pMass = mass;

    __asm__ volatile (
        // Загружаем и размножаем скаляры на все 4 линии регистров v4, v5, v6
        "ld1r {v4.4s}, [%[drag]]\n\t"
        "ld1r {v5.4s}, [%[ax]]\n\t"
        "ld1r {v6.4s}, [%[ay]]\n\t"

        "1:\n\t"
        // Пакетная загрузка 4 элементов с аппаратным сдвигом указателя на 16 байт
        "ld1 {v7.4s}, [%[act]], #16\n\t"     // v7 = active[i..i+3]
        "ld1 {v2.4s}, [%[ms]],  #16\n\t"     // v2 = mass[i..i+3]

        // Скорость X: vx = (vx * drag + ax * mass) * active
        "ld1  {v0.4s}, [%[vx]]\n\t"
        "fmul v0.4s, v0.4s, v4.4s\n\t"       // v0 = vx * drag
        "fmla v0.4s, v5.4s, v2.4s\n\t"       // v0 += ax * mass (за 1 такт через FMLA!)
        "fmul v0.4s, v0.4s, v7.4s\n\t"       // Branchless: если в сугробе (0.0), vx = 0
        "st1  {v0.4s}, [%[vx]], #16\n\t"

        // Скорость Y: vy = (vy * drag + ay * mass) * active
        "ld1  {v1.4s}, [%[vy]]\n\t"
        "fmul v1.4s, v1.4s, v4.4s\n\t"       // v1 = vy * drag
        "fmla v1.4s, v6.4s, v2.4s\n\t"       // v1 += ay * mass
        "fmul v1.4s, v1.4s, v7.4s\n\t"       // Branchless: если в сугробе (0.0), vy = 0
        "st1  {v1.4s}, [%[vy]], #16\n\t"

        // Координаты: px += vx, py += vy
        "ld1  {v2.4s}, [%[px]]\n\t"
        "fadd v2.4s, v2.4s, v0.4s\n\t"
        "st1  {v2.4s}, [%[px]], #16\n\t"

        "ld1  {v3.4s}, [%[py]]\n\t"
        "fadd v3.4s, v3.4s, v1.4s\n\t"
        "st1  {v3.4s}, [%[py]], #16\n\t"

        "subs %[cnt], %[cnt], #1\n\t"
        "b.ne 1b\n\t"
        : [px] "+r"(pPx), [py] "+r"(pPy),
          [vx] "+r"(pVx), [vy] "+r"(pVy),
          [act] "+r"(pAct), [ms] "+r"(pMass),
          [cnt] "+r"(loops)
        : [drag] "r"(&drag), [ax] "r"(&ax), [ay] "r"(&ay)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory", "cc"
    );
#else
    // Резервный цикл на случай запуска на 32-битном ARM или ПК
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        float act = active[i];
        float m   = mass[i];
        vx[i] = (vx[i] * drag + ax * m) * act;
        vy[i] = (vy[i] * drag + ay * m) * act;
        px[i] += vx[i];
        py[i] += vy[i];
    }
#endif
}

// ============================================================================
// УПРУГИЕ СТОЛКНОВЕНИЯ СНЕЖИНОК В ВОЗДУХЕ ЧЕРЕЗ SPATIAL GRID O(N)
// ============================================================================
void ResolveMidAirCollisions() {
    std::memset(cellHead, -1, sizeof(cellHead));

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        if (active[i] == 0.0f) continue;
        int cx = ((int)px[i]) >> CELL_SHIFT;
        int cy = ((int)py[i]) >> CELL_SHIFT;
        if (cx >= 1 && cx < GRID_W - 1 && cy >= 1 && cy < GRID_H - 1) {
            int cellIdx = cy * GRID_W + cx;
            nextParticle[i] = cellHead[cellIdx];
            cellHead[cellIdx] = i;
        }
    }

    constexpr float minDist = 2.2f;
    constexpr float minDist2 = minDist * minDist;
    constexpr float restitution = 0.55f;
    const int neighborOffsets[5] = { 0, 1, GRID_W - 1, GRID_W, GRID_W + 1 };

    for (int cy = 1; cy < GRID_H - 2; ++cy) {
        for (int cx = 1; cx < GRID_W - 2; ++cx) {
            int baseCell = cy * GRID_W + cx;
            int i = cellHead[baseCell];

            while (i != -1) {
                for (int n = 0; n < 5; ++n) {
                    int j = (n == 0) ? nextParticle[i] : cellHead[baseCell + neighborOffsets[n]];
                    while (j != -1) {
                        float dx = px[j] - px[i];
                        float dy = py[j] - py[i];
                        float d2 = dx * dx + dy * dy;

                        if (d2 < minDist2 && d2 > 0.0001f) {
                            float dist = std::sqrt(d2);
                            float nx = dx / dist;
                            float ny = dy / dist;

                            // Раздвигаем только в свободные клетки воды (не вдавливая в ёлку)
                            float overlap = (minDist - dist) * 0.4f;
                            float nxi = px[i] - nx * overlap;
                            float nyi = py[i] - ny * overlap;
                            if (nxi > 2 && nxi < W - 2 && nyi > 2 && nyi < H - 2 &&
                                snowGrid[(int)nyi][(int)nxi] == 0) {
                                px[i] = nxi;
                                py[i] = nyi;
                            }

                            float nxj = px[j] + nx * overlap;
                            float nyj = py[j] + ny * overlap;
                            if (nxj > 2 && nxj < W - 2 && nyj > 2 && nyj < H - 2 &&
                                snowGrid[(int)nyj][(int)nxj] == 0) {
                                px[j] = nxj;
                                py[j] = nyj;
                            }

                            float rvx = vx[j] - vx[i];
                            float rvy = vy[j] - vy[i];
                            float velAlongNormal = rvx * nx + rvy * ny;

                            if (velAlongNormal < 0.0f) {
                                float invM1 = 1.0f / mass[i];
                                float invM2 = 1.0f / mass[j];
                                float impulse = -(1.0f + restitution) * velAlongNormal / (invM1 + invM2);
                                vx[i] -= impulse * invM1 * nx;
                                vy[i] -= impulse * invM1 * ny;
                                vx[j] += impulse * invM2 * nx;
                                vy[j] += impulse * invM2 * ny;
                            }
                        }
                        j = nextParticle[j];
                    }
                }
                i = nextParticle[i];
            }
        }
    }
}

void InitGlobe() {
    int baseY = GLOBE_CY + RADIUS - 12;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int dx = x - GLOBE_CX;
            int dy = y - GLOBE_CY;
            snowGrid[y][x] = (dx * dx + dy * dy >= RADIUS * RADIUS || y >= baseY) ? 3 : 0;
        }
    }

    // Ствол и три яруса веток ёлки
    for (int y = baseY - 26; y < baseY; ++y)
        for (int x = GLOBE_CX - 6; x <= GLOBE_CX + 6; ++x) snowGrid[y][x] = 2;

    for (int tier = 0; tier < 3; ++tier) {
        int topY = baseY - 125 + tier * 32;
        int botY = topY + 48;
        for (int y = topY; y < botY; ++y) {
            int halfW = (y - topY) * (28 + tier * 8) / 48;
            for (int x = GLOBE_CX - halfW; x <= GLOBE_CX + halfW; ++x) snowGrid[y][x] = 2;
        }
    }

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        while (true) {
            float angle = RandFloat(0.0f, 6.283185f);
            float r = std::sqrt(RandFloat(0.0f, 0.92f)) * (RADIUS - 8);
            int ix = (int)(GLOBE_CX + std::cos(angle) * r);
            int iy = (int)(GLOBE_CY + std::sin(angle) * r);
            if (ix > 4 && ix < W - 4 && iy > 4 && iy < H - 4 && snowGrid[iy][ix] == 0) {
                px[i] = (float)ix;
                py[i] = (float)iy;
                break;
            }
        }
        vx[i] = RandFloat(-1.5f, 1.5f);
        vy[i] = RandFloat(-1.5f, 1.5f);
        active[i] = 1.0f;
        mass[i] = RandFloat(0.45f, 1.35f);
    }
}

void ResolveBoundariesAndSettling(float shakeIntensity, float shakeVx, float shakeVy, int frame) {
    const float maxR = (float)(RADIUS - 2);
    const float maxR2 = maxR * maxR;

    // Срыв осевших сугробов при встряхивании телефона или взмахе пальцем
    if (shakeIntensity > 0.35f) {
        uint32_t wakeProb = (uint32_t)(shakeIntensity * 18000.0f);
        if (wakeProb > 65535) wakeProb = 65535;
        for (int i = 0; i < NUM_PARTICLES; ++i) {
            if (active[i] == 0.0f && (FastRand() & 0xFFFF) < wakeProb) {
                int ix = (int)px[i], iy = (int)py[i];
                if (ix >= 0 && ix < W && iy >= 0 && iy < H && snowGrid[iy][ix] == 1)
                    snowGrid[iy][ix] = 0;
                active[i] = 1.0f;
                float dx = px[i] - GLOBE_CX, dy = py[i] - GLOBE_CY;
                vx[i] = shakeVx * RandFloat(0.4f, 1.3f) - dy * 0.015f + RandFloat(-2.2f, 2.2f);
                vy[i] = shakeVy * RandFloat(0.4f, 1.3f) + dx * 0.015f - RandFloat(1.0f, 4.2f);
            }
        }
    }

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        if (active[i] == 0.0f) continue;

        vx[i] += std::sin(py[i] * 0.06f + frame * 0.04f + (i & 15)) * 0.012f;

        // Отскок от стеклянной сферы
        float dx = px[i] - GLOBE_CX, dy = py[i] - GLOBE_CY;
        float dist2 = dx * dx + dy * dy;
        if (dist2 >= maxR2) {
            float dist = std::sqrt(dist2);
            float nx = dx / dist, ny = dy / dist;
            px[i] = GLOBE_CX + nx * (maxR - 1.0f);
            py[i] = GLOBE_CY + ny * (maxR - 1.0f);
            float dot = vx[i] * nx + vy[i] * ny;
            if (dot > 0.0f) {
                vx[i] = (vx[i] - 1.4f * dot * nx) * 0.7f;
                vy[i] = (vy[i] - 1.4f * dot * ny) * 0.7f;
            }
        }

        int ix = (int)px[i], iy = (int)py[i];
        if (ix < 2 || ix >= W - 2 || iy < 2 || iy >= H - 2) continue;

        // Защита от «эффекта лифта» внутри ёлки
        if (snowGrid[iy][ix] != 0) {
            int prevX = (int)(px[i] - vx[i]);
            int prevY = (int)(py[i] - vy[i]);
            if (prevX > 2 && prevX < W - 2 && prevY > 2 && prevY < H - 2 && snowGrid[prevY][prevX] == 0) {
                px[i] -= vx[i];
                py[i] -= vy[i];
                vx[i] *= -0.35f;
                vy[i] *= 0.2f;
                ix = (int)px[i];
                iy = (int)py[i];
            } else if (snowGrid[iy - 1][ix] == 0) {
                py[i] = (float)(iy - 1);
                vy[i] = 0.0f;
                iy -= 1;
            } else {
                float pushDir = (px[i] < (float)GLOBE_CX) ? -1.5f : 1.5f;
                px[i] += pushDir;
                vx[i] = pushDir * 0.3f;
                if (vy[i] < 0.0f) vy[i] = 0.0f;
                continue;
            }
        }

        // Осыпание по склонам и накопление сугробов
        if (vy[i] >= -0.05f && snowGrid[iy + 1][ix] != 0) {
            int dir = (FastRand() & 1) ? 1 : -1;
            if (snowGrid[iy][ix + dir] == 0 && snowGrid[iy + 1][ix + dir] == 0) {
                px[i] += dir * 0.85f;
                vx[i] = dir * 0.25f;
            } else if (snowGrid[iy][ix - dir] == 0 && snowGrid[iy + 1][ix - dir] == 0) {
                px[i] -= dir * 0.85f;
                vx[i] = -dir * 0.25f;
            } else {
                float speed2 = vx[i] * vx[i] + vy[i] * vy[i];
                if (speed2 < 2.2f && shakeIntensity < 0.6f && snowGrid[iy][ix] == 0) {
                    snowGrid[iy][ix] = 1;
                    active[i] = 0.0f;
                    vx[i] = 0.0f;
                    vy[i] = 0.0f;
                    px[i] = (float)ix;
                    py[i] = (float)iy;
                } else {
                    vy[i] = 0.0f;
                    vx[i] *= 0.5f;
                }
            }
        }
    }
}

void RenderFrame(bool buttonPressed) {
    int baseY = GLOBE_CY + RADIUS - 12;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int dx = x - GLOBE_CX, dy = y - GLOBE_CY;
            int d2 = dx * dx + dy * dy;

            if (d2 <= RADIUS * RADIUS && y < baseY) {
                int r = 10 + (y * 18) / H;
                int g = 20 + (y * 32) / H;
                int b = 45 + (y * 55) / H;
                int gx = dx + 75, gy = dy + 75;
                if (gx * gx + gy * gy < 3200) { r += 18; g += 24; b += 32; }
                if (d2 > (RADIUS - 3) * (RADIUS - 3)) { r = 150; g = 185; b = 220; }
                pixels[y * W + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            } else if (y >= baseY && y < baseY + 42 && std::abs(dx) < 155 + (y - baseY) / 2) {
                int shade = 100 - std::abs(dx) / 3;
                pixels[y * W + x] = 0xFF000000 | ((shade + 25) << 16) | ((shade - 10) << 8) | 30;
            } else if (y >= H - 56 && y <= H - 14 && std::abs(dx) <= 130) {
                // Сенсорная кнопка «ВСТРЯХНУТЬ» внизу экрана
                uint32_t btnColor = buttonPressed ? 0xFF3B82F6 : 0xFF1E3A5F;
                if (y == H - 56 || y == H - 14 || std::abs(dx) == 130) btnColor = 0xFF7DD3FC;
                pixels[y * W + x] = btnColor;
            } else {
                pixels[y * W + x] = 0xFF12131A;
            }

            if (snowGrid[y][x] == 2) {
                pixels[y * W + x] = (y >= baseY - 26 && std::abs(dx) <= 6)
                    ? 0xFF6B4423 : (((x + y) & 1) ? 0xFF1E6B3A : 0xFF16522C);
            }
        }
    }

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        int ix = (int)px[i], iy = (int)py[i];
        if (ix >= 1 && ix < W - 1 && iy >= 1 && iy < H - 1) {
            if (active[i] == 0.0f) {
                pixels[iy * W + ix] = (iy & 1) ? 0xFFF4F8FF : 0xFFDCE8F7;
            } else {
                uint32_t col = (mass[i] > 0.9f) ? 0xFFFFFFFF : 0xFFB8D0EC;
                pixels[iy * W + ix] = col;
                if (mass[i] > 1.15f) pixels[iy * W + ix + 1] = col;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_SENSOR);

    SDL_Window* window = SDL_CreateWindow(
        "Snow Globe ARM64 NEON",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        W, H, SDL_WINDOW_SHOWN | SDL_WINDOW_FULLSCREEN_DESKTOP
    );
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    SDL_RenderSetLogicalSize(renderer, W, H);

    SDL_Texture* texture = SDL_CreateTexture(
        renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, W, H
    );

    // Подключаем встроенный акселерометр смартфона Android
    SDL_Sensor* accelSensor = nullptr;
    int numSensors = SDL_NumSensors();
    for (int i = 0; i < numSensors; ++i) {
        if (SDL_SensorGetDeviceType(i) == SDL_SENSOR_ACCEL) {
            accelSensor = SDL_SensorOpen(i);
            break;
        }
    }

    InitGlobe();

    float prevAccel[3] = { 0.0f, 9.8f, 0.0f };
    bool accelInitialized = false;

    float touchShakeVx = 0.0f, touchShakeVy = 0.0f;
    bool buttonPressed = false;
    int frame = 0;
    bool running = true;

    while (running) {
        touchShakeVx *= 0.6f;
        touchShakeVy *= 0.6f;

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) {
                running = false;
            } else if (ev.type == SDL_FINGERMOTION) {
                // Взмах пальцем по экрану закручивает воду и снег
                touchShakeVx += ev.tfinger.dx * W * 0.45f;
                touchShakeVy += ev.tfinger.dy * H * 0.45f;
            } else if (ev.type == SDL_FINGERDOWN) {
                float tx = ev.tfinger.x * W;
                float ty = ev.tfinger.y * H;
                // Если нажали на нижнюю кнопку или тапнули по шару
                if (ty >= H - 70 && std::abs(tx - GLOBE_CX) <= 140) {
                    buttonPressed = true;
                } else {
                    touchShakeVx += RandFloat(-8.0f, 8.0f);
                    touchShakeVy -= RandFloat(5.0f, 10.0f);
                }
            } else if (ev.type == SDL_FINGERUP) {
                buttonPressed = false;
            } else if (ev.type == SDL_MOUSEMOTION && (ev.motion.state & SDL_BUTTON_LMASK)) {
                touchShakeVx += ev.motion.xrel * 0.35f;
                touchShakeVy += ev.motion.yrel * 0.35f;
            }
        }

        if (buttonPressed) {
            touchShakeVx += RandFloat(-12.0f, 12.0f);
            touchShakeVy -= RandFloat(8.0f, 16.0f);
        }

        // Читаем физическую тряску и наклон самого телефона с акселерометра
        float sensorShakeVx = 0.0f, sensorShakeVy = 0.0f;
        float gravityX = 0.0f, gravityY = 0.042f;

        if (accelSensor != nullptr) {
            float curAccel[3] = { 0.0f, 0.0f, 0.0f };
            if (SDL_SensorGetData(accelSensor, curAccel, 3) == 0) {
                if (!accelInitialized) {
                    std::memcpy(prevAccel, curAccel, sizeof(curAccel));
                    accelInitialized = true;
                }
                // Рывок (производная ускорения) при тряске смартфона в руке
                float jerkX = curAccel[0] - prevAccel[0];
                float jerkY = curAccel[1] - prevAccel[1];
                std::memcpy(prevAccel, curAccel, sizeof(curAccel));

                if (std::fabs(jerkX) > 1.2f || std::fabs(jerkY) > 1.2f) {
                    sensorShakeVx = -jerkX * 1.8f;
                    sensorShakeVy =  jerkY * 1.8f;
                }
                // Наклон телефона меняет вектор гравитации для снежинок
                gravityX = -curAccel[0] * 0.0045f;
                gravityY =  curAccel[1] * 0.0045f;
                if (gravityY < 0.01f && std::fabs(curAccel[1]) < 2.0f) gravityY = 0.035f;
            }
        }

        float totalShakeVx = touchShakeVx + sensorShakeVx;
        float totalShakeVy = touchShakeVy + sensorShakeVy;
        float shakeIntensity = std::sqrt(totalShakeVx * totalShakeVx + totalShakeVy * totalShakeVy);

        // 1. Векторный шаг физики на мобильном ассемблере ARM64 NEON
        StepPhysicsASM(0.985f, gravityX + totalShakeVx * 0.08f, gravityY + totalShakeVy * 0.08f);

        // 2. Упругие столкновения снежинок в воздухе через Spatial Grid
        ResolveMidAirCollisions();

        // 3. Оседание в сугробы и границы шара
        ResolveBoundariesAndSettling(shakeIntensity, totalShakeVx, totalShakeVy, frame++);

        // 4. Отрисовка кадра через аппаратную текстуру SDL2
        RenderFrame(buttonPressed);
        SDL_UpdateTexture(texture, nullptr, pixels, W * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

    if (accelSensor) SDL_SensorClose(accelSensor);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
  
