#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <windowsx.h>
#include <tchar.h>

#include <memory>

#include "GameParticle.h"

#define MAX_LOADSTRING 100

WCHAR szTitle[MAX_LOADSTRING] = L"Particle Game CUDA";
WCHAR szWindowClass[MAX_LOADSTRING] = L"ParticleGameCUDA";

ATOM MyRegisterClass(HINSTANCE hInstance);
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow);
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

std::unique_ptr<GameParticle> g_GameParticle;

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR lpCmdLine,
    _In_ int nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    MyRegisterClass(hInstance);

    if (!InitInstance(hInstance, nCmdShow))
    {
        OutputDebugString(_T("Error in InitInstance\n"));
        return FALSE;
    }

    MSG msg{};
    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else if (g_GameParticle)
            g_GameParticle->processFrame();
    }

    if (g_GameParticle)
    {
        g_GameParticle->shutdownGame();
        g_GameParticle.reset();
    }

    return static_cast<int>(msg.wParam);
}

ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex{};
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.hInstance = hInstance;
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = nullptr;
    wcex.lpszClassName = szWindowClass;

    return RegisterClassExW(&wcex);
}

BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
    constexpr int clientWidth = 1280;
    constexpr int clientHeight = 720;

    RECT rect{ 0, 0, clientWidth, clientHeight };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    HWND hWnd = CreateWindowW(
        szWindowClass,
        szTitle,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        0,
        rect.right - rect.left,
        rect.bottom - rect.top,
        nullptr,
        nullptr,
        hInstance,
        nullptr
    );

    if (!hWnd)
    {
        OutputDebugString(_T("Error in CreateWindow\n"));
        return FALSE;
    }

    g_GameParticle = std::make_unique<GameParticle>();
    if (!g_GameParticle->initializeGame(hWnd, clientWidth, clientHeight))
    {
        MessageBoxW(hWnd, L"Game/DirectX initialization failed. Check DirectX 11 and CUDA setup.", L"Particle Game CUDA", MB_OK | MB_ICONERROR);
        g_GameParticle.reset();
        return FALSE;
    }

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    return TRUE;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_SIZE:
        if (g_GameParticle && wParam != SIZE_MINIMIZED)
            g_GameParticle->handleWindowResize(hWnd);
        return 0;

    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
        if (g_GameParticle)
            g_GameParticle->OnMouseButtonDown(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
        break;

    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
        if (g_GameParticle)
            g_GameParticle->OnMouseButtonUp(wParam);
        break;

    case WM_MOUSEMOVE:
        if (g_GameParticle)
            g_GameParticle->OnMouseMove(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
        break;

    case WM_MOUSEWHEEL:
        if (g_GameParticle)
            g_GameParticle->OnMouseWheel(GET_WHEEL_DELTA_WPARAM(wParam));
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
}