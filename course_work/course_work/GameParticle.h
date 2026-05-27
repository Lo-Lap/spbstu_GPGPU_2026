#ifndef GAME_PARTICLE
#define GAME_PARTICLE

#include <chrono>
#include <string>

#include <windows.h>

#include "SceneTypes.h"
#include "ParticleEngine.cuh"
#include "Render.h"

struct FrameTime
{
    float dt;
    float elapsed;
};

class GameParticle
{
public:
    GameParticle();
    ~GameParticle();

    bool initializeGame(HWND hwnd, int width, int height);
    void processFrame();
    void handleWindowResize(HWND hwnd);
    void shutdownGame();
    bool isRunning() const { return running; }

    void OnMouseButtonDown(WPARAM wParam, int x, int y);
    void OnMouseButtonUp(WPARAM wParam);
    void OnMouseMove(int x, int y);
    void OnMouseWheel(short delta);

private:
    enum class SelectedObject
    {
        Sphere,
        //Box,
        Plane1,
        Plane2
    };

    SimulationParameters settings{};
    GridInfo grid{};
    ParticleEmitter emitter{};
    CollisionScene scene{};
    ParticleEngine particleEngine;
    Render render;

    SelectedObject selectedObject = SelectedObject::Sphere;
    bool running = true;
    bool paused = false;
    bool previousPKeyDown = false;
    bool initialized = false;
    bool gameOverMessageShown = false;

    std::chrono::high_resolution_clock::time_point startTime{};
    std::chrono::high_resolution_clock::time_point lastFrameTime{};

    POINT lastMousePos{};
    bool leftMouseDown = false;
    bool rightMouseDown = false;
    float mouseMoveSensitivity = 0.02f;
    float mouseRotateSensitivity = 0.01f;
    float wheelSensitivity = 0.1f;

    FrameTime computeDeltaTime();

    void processInput(float dt);
    void updateParticleSimulation(float dt);
    void renderCurrentFrame(float elapsed, int score);
    void checkEndCondition(float elapsed, int score);

    void moveSelectedObject(float dx, float dy, float dz);
    void updateWindowTitleText(float elapsedSeconds, int score);
    std::string getSelectedObjectName() const;

    void handleTranslation(float dx, float dy);
    void handleRotation(float dx, float dy);
    void handleWheelScroll(short delta);
};
#endif