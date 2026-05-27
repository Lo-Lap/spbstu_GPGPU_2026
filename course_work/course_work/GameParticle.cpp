#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "GameParticle.h"
#include "CudaCheck.h"
#include "VectorMath.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

SimulationParameters initParameters()
{
    SimulationParameters s{};
    s.particleCount = 20000;
    s.targetScore = 1200;
    s.gameDuration = 15.0f;
    s.particleRadius = 0.08f;
    s.worldMinY = -18.0f;
    s.damping = 0.98f;
    s.rollingFriction = 0.985f;
    s.gravity = 9.8f;
    return s;
}

GridInfo initGrid()
{
    GridInfo grid{};
    grid.origin = make_float3(-16.0f, -20.0f, -16.0f);
    grid.cellSize = 1.0f;
    grid.dims = make_int3(32, 32, 32);
    return grid;
}

ParticleEmitter initEmitter()
{
    ParticleEmitter e{};
    e.center = make_float3(0.0f, 12.0f, 0.0f);
    e.size = make_float3(5.5f, 0.5f, 5.5f);
    e.baseVelocity = make_float3(0.0f, -4.0f, 0.0f);
    return e;
}

CollisionScene initScene()
{
    CollisionScene scene{};

    scene.sphere.center = make_float3(3.0f, -1.0f, 0.0f);
    scene.sphere.radius = 2.35f;

    scene.planeCount = 2;

    scene.planes[0].point = make_float3(-3.0f, -3.0f, 0.0f);
    scene.planes[0].normal = normalize3(make_float3(0.45f, 1.0f, 0.15f));
    scene.planes[0].halfSize = 5.0f;

    scene.planes[1].point = make_float3(3.0f, -8.0f, 0.0f);
    scene.planes[1].normal = normalize3(make_float3(-0.45f, 1.0f, -0.15f));
    scene.planes[1].halfSize = 5.0f;

    scene.basket.minCorner = make_float3(1.0f, -16.0f, -2.0f);
    scene.basket.maxCorner = make_float3(5.0f, -14.0f, 2.0f);

    return scene;
}

static const float3 kCameraEye = make_float3(11.5f, 3.5f, -28.0f);
static const float3 kCameraAt = make_float3(0.0f, -3.0f, 0.0f);
static const float3 kCameraUpWorld = make_float3(0.0f, 1.0f, 0.0f);
static float3 g_cameraRight;
static float3 g_cameraUp;
static float3 g_cameraLook;

static void computeCameraVectors()
{
    g_cameraLook = normalize3(make_float3(kCameraAt.x - kCameraEye.x,
        kCameraAt.y - kCameraEye.y,
        kCameraAt.z - kCameraEye.z));
    g_cameraRight = normalize3(cross3(kCameraUpWorld, g_cameraLook));
    g_cameraUp = cross3(g_cameraRight, g_cameraLook);
}

static float3 rotateAroundAxis(const float3& v, const float3& axis, float angle)
{
    float c = cosf(angle);
    float s = sinf(angle);
    float dot = dot3(axis, v);
    float3 cross = cross3(axis, v);
    return make_float3(
        v.x * c + cross.x * s + axis.x * dot * (1 - c),
        v.y * c + cross.y * s + axis.y * dot * (1 - c),
        v.z * c + cross.z * s + axis.z * dot * (1 - c)
    );
}

GameParticle::GameParticle(): settings(initParameters()),grid(initGrid()),emitter(initEmitter()),scene(initScene()),
    particleEngine(settings.particleCount, grid)
{
    particleEngine.initialize(emitter, 1337UL);
    computeCameraVectors();
}

GameParticle::~GameParticle()
{
    shutdownGame();
}

bool GameParticle::initializeGame(HWND hwnd, int width, int height)
{
    if (!render.initializeRender(hwnd, width, height))
        return false;

    startTime = std::chrono::high_resolution_clock::now();
    lastFrameTime = startTime;
    initialized = true;
    return true;
}

void GameParticle::processFrame()
{
    if (!initialized || !running) 
        return;

    auto [dt, elapsed] = computeDeltaTime();
    processInput(dt);

    if (!paused)
        updateParticleSimulation(dt);

    int score = particleEngine.getScore();
    renderCurrentFrame(elapsed, score);
    checkEndCondition(elapsed, score);
}

FrameTime GameParticle::computeDeltaTime()
{
    using clock = std::chrono::high_resolution_clock;
    auto now = clock::now();
    float dt = std::min(std::chrono::duration<float>(now - lastFrameTime).count(), 1.0f / 30.0f);
    float elapsed = std::chrono::duration<float>(now - startTime).count();
    lastFrameTime = now;
    return { dt, elapsed };
}

void GameParticle::updateParticleSimulation(float dt)
{
    particleEngine.step(dt, scene, settings, emitter);
}

void GameParticle::renderCurrentFrame(float elapsed, int score)
{
    auto state = particleEngine.downloadParticleState(settings.particleCount);
    render.renderFrame(scene, state, settings.particleRadius);
    updateWindowTitleText(elapsed, score);
}

void GameParticle::checkEndCondition(float elapsed, int score)
{
    if (gameOverMessageShown)
        return;

    if (score >= settings.targetScore)
    {
        paused = true;
        render.setWindowTitle("Victory! Basket filled. Press Esc to exit.");
        MessageBoxW(GetActiveWindow(),
            L"Корзина наполнена\nНажмите Esc для выхода.",
            L"Победа", MB_OK | MB_ICONINFORMATION);
        gameOverMessageShown = true;
    }
    else if (elapsed >= settings.gameDuration)
    {
        paused = true;
        render.setWindowTitle("Time is over. Press Esc to exit.");
        MessageBoxW(GetActiveWindow(),
            L"Время истекло. Нужное количество частиц не набрано.\nНажмите Esc для выхода.",
            L"Игра окончена", MB_OK | MB_ICONINFORMATION);
        gameOverMessageShown = true;
    }
}

void GameParticle::handleWindowResize(HWND hwnd)
{
    render.handleResize(hwnd);
}

void GameParticle::shutdownGame()
{
    if (initialized)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        render.shutdownRender();
        initialized = false;
    }
}

void GameParticle::processInput(float)
{
    if (render.isKeyPressed(KEY_ESCAPE))
    {
        running = false;
        PostQuitMessage(0);
        return;
    }

    if (render.isKeyPressed(KEY_1)) 
        selectedObject = SelectedObject::Sphere;

    if (render.isKeyPressed(KEY_2))
        selectedObject = SelectedObject::Plane1;

    if (render.isKeyPressed(KEY_3)) 
        selectedObject = SelectedObject::Plane2;

    bool pDown = render.isKeyPressed(KEY_P);
    if (pDown && !previousPKeyDown)
        paused = !paused;
    previousPKeyDown = pDown;
}

void GameParticle::OnMouseButtonDown(WPARAM wParam, int x, int y)
{
    if (wParam & MK_LBUTTON)
        leftMouseDown = true;
    if (wParam & MK_RBUTTON) 
        rightMouseDown = true;

    lastMousePos.x = x;
    lastMousePos.y = y;
}

void GameParticle::OnMouseButtonUp(WPARAM wParam)
{
    if (!(wParam & MK_LBUTTON)) 
        leftMouseDown = false;

    if (!(wParam & MK_RBUTTON)) 
        rightMouseDown = false;
}

void GameParticle::OnMouseMove(int x, int y)
{
    POINT current{ x, y };

    if (!leftMouseDown && !rightMouseDown)
    {
        lastMousePos = current;
        return;
    }

    float dx = static_cast<float>(current.x - lastMousePos.x);
    float dy = static_cast<float>(current.y - lastMousePos.y);
    lastMousePos = current;

    if (fabs(dx) < 0.5f && fabs(dy) < 0.5f)
        return;

    if (leftMouseDown)
        handleTranslation(dx, dy);

    if (rightMouseDown)
        handleRotation(dx, dy);
}

void GameParticle::OnMouseWheel(short delta)
{
    handleWheelScroll(delta);
}

void GameParticle::handleTranslation(float dx, float dy)
{
    float3 move;
    move.x = g_cameraRight.x * dx * mouseMoveSensitivity - g_cameraUp.x * dy * mouseMoveSensitivity;
    move.y = g_cameraRight.y * dx * mouseMoveSensitivity - g_cameraUp.y * dy * mouseMoveSensitivity;
    move.z = g_cameraRight.z * dx * mouseMoveSensitivity - g_cameraUp.z * dy * mouseMoveSensitivity;

    moveSelectedObject(move.x, move.y, move.z);
}

void GameParticle::handleRotation(float dx, float dy)
{
    float3* normalPtr = nullptr;
    if (selectedObject == SelectedObject::Plane1)
        normalPtr = &scene.planes[0].normal;
    else if (selectedObject == SelectedObject::Plane2)
        normalPtr = &scene.planes[1].normal;
    else
        return;

    float angleX = dx * mouseRotateSensitivity;
    float angleY = dy * mouseRotateSensitivity;

    float3& n = *normalPtr;
    n = rotateAroundAxis(n, make_float3(0.0f, 1.0f, 0.0f), angleX);
    n = rotateAroundAxis(n, g_cameraRight, angleY);

    n.y = std::max(0.25f, n.y);
    n = normalize3(n);
}

void GameParticle::handleWheelScroll(short delta)
{
    float amount = (delta / 120.0f) * wheelSensitivity;
    float3 move;
    move.x = g_cameraLook.x * amount;
    move.y = g_cameraLook.y * amount;
    move.z = g_cameraLook.z * amount;
    moveSelectedObject(move.x, move.y, move.z);
}

void GameParticle::moveSelectedObject(float dx, float dy, float dz)
{
    switch (selectedObject)
    {
    case SelectedObject::Sphere:
        scene.sphere.center.x += dx;
        scene.sphere.center.y += dy;
        scene.sphere.center.z += dz;
        break;
   /* case SelectedObject::Box:
        scene.box.center.x += dx;
        scene.box.center.y += dy;
        scene.box.center.z += dz;
        break;*/
    case SelectedObject::Plane1:
        scene.planes[0].point.x += dx;
        scene.planes[0].point.y += dy;
        scene.planes[0].point.z += dz;
        break;
    case SelectedObject::Plane2:
        scene.planes[1].point.x += dx;
        scene.planes[1].point.y += dy;
        scene.planes[1].point.z += dz;
        break;
    }
}

void GameParticle::updateWindowTitleText(float elapsedSeconds, int score)
{
    float timeLeft = std::max(0.0f, settings.gameDuration - elapsedSeconds);
    std::ostringstream oss;
    oss << "Particle Game CUDA | Time: " << static_cast<int>(timeLeft + 0.5f)
        << "s | Score: " << score << "/" << settings.targetScore
        << " | Selected: " << getSelectedObjectName()
        << " | " << (paused ? "paused" : "running");

    std::string title = oss.str();
    render.setWindowTitle(title);
}

std::string GameParticle::getSelectedObjectName() const
{
    switch (selectedObject)
    {
    case SelectedObject::Sphere: return "sphere";
    //case SelectedObject::Box:    return "box";
    case SelectedObject::Plane1: return "plane1";
    case SelectedObject::Plane2: return "plane2";
    }
    return "unknown";
}