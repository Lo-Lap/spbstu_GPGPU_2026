#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "Render.h"
#include "VectorMath.cuh"

#include <dxgi.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace DirectX;

constexpr float kPi = 3.14159265358979323846f;

constexpr int kSphereRings = 6;
constexpr int kSphereSegments = 128;
constexpr int kFloorCells = 16;
constexpr float kFloorY = -16.0f;
constexpr float kFloorExtent = 16.0f;

constexpr float kFloorMajorColor[3] = { 0.18f, 0.18f, 0.22f };
constexpr float kPlaneBorderColor[3] = { 0.50f, 0.50f, 0.55f };
constexpr float kPlaneGridColor[3] = { 0.20f, 0.20f, 0.24f };
constexpr float kSphereColor[3] = { 0.95f, 0.65f, 0.15f };
constexpr float kBoxColor[3] = { 0.20f, 0.75f, 0.90f };
constexpr float kBasketColor[3] = { 0.90f, 0.25f, 0.40f };

struct SceneConstantBuffer 
{
    XMMATRIX mvp;
    XMFLOAT2 invViewport;
    float pointSize;
    float padding;
};

template <typename T>
void safeRelease(T*& ptr) 
{
    if (ptr) 
    {
        ptr->Release();
        ptr = nullptr;
    }
}

Vertex makeVertex(float3 p, float r, float g, float b, float a = 1.0f) 
{
    return Vertex{ p.x, p.y, p.z, r, g, b, a };
}

void addLine(std::vector<Vertex>& lines, float3 a, float3 b, float r, float g, float bl) 
{
    lines.push_back(makeVertex(a, r, g, bl));
    lines.push_back(makeVertex(b, r, g, bl));
}

std::string hresultToString(HRESULT hr) 
{
    std::ostringstream oss;
    oss << "HRESULT 0x" << std::hex << static_cast<unsigned long>(hr);
    return oss.str();
}

std::wstring shaderPath(const wchar_t* fileName) 
{
    std::wstring p = L"shaders/";
    p += fileName;
    return p;
}


Render::~Render() 
{
    releaseAllResources();
}

bool Render::initializeRender(HWND hwnd, int width, int height) 
{
    this->hwnd = hwnd;
    this->width = width;
    this->height = height;

    if (!createDeviceAndSwapChain()) 
    {
        return false;
    }
    if (!createRenderTargets()) 
    {
        return false;
    }
    if (!createShadersAndInputLayout()) 
    {
        return false;
    }
    if (!createRasterizerDepthBlendStates()) 
    {
        return false;
    }
    if (!createConstantBuffer()) 
    {
        return false;
    }
    return true;
}

bool Render::createDeviceAndSwapChain() 
{
    HRESULT result = S_OK;

    IDXGIFactory* factory = nullptr;
    result = CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&factory));
    if (FAILED(result)) 
    {
        std::cerr << "CreateDXGIFactory failed: " << hresultToString(result) << "\n";
        return false;
    }

    IDXGIAdapter* selectedAdapter = nullptr;
    IDXGIAdapter* adapter = nullptr;
    UINT adapterIndex = 0;
    while (factory->EnumAdapters(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND) 
    {
        DXGI_ADAPTER_DESC desc{};
        adapter->GetDesc(&desc);
        if (wcscmp(desc.Description, L"Microsoft Basic Render Driver") != 0) 
        {
            selectedAdapter = adapter;
            break;
        }
        adapter->Release();
        adapter = nullptr;
        ++adapterIndex;
    }

    UINT flags = 0;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL level{};
    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0 };

    result = D3D11CreateDevice(
        selectedAdapter,
        selectedAdapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        levels,
        1,
        D3D11_SDK_VERSION,
        &device,
        &level,
        &context
    );

#ifdef _DEBUG
    if (FAILED(result) && (flags & D3D11_CREATE_DEVICE_DEBUG)) 
    {
        flags &= ~D3D11_CREATE_DEVICE_DEBUG;
        result = D3D11CreateDevice(
            selectedAdapter,
            selectedAdapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            flags,
            levels,
            1,
            D3D11_SDK_VERSION,
            &device,
            &level,
            &context
        );
    }
#endif

    if (SUCCEEDED(result) && context) 
    {
        context->QueryInterface(__uuidof(ID3DUserDefinedAnnotation), reinterpret_cast<void**>(&annotation));
    }

    if (SUCCEEDED(result)) 
    {
        DXGI_SWAP_CHAIN_DESC swapDesc{};
        swapDesc.BufferCount = 2;
        swapDesc.BufferDesc.Width = static_cast<UINT>(std::max(1, width));
        swapDesc.BufferDesc.Height = static_cast<UINT>(std::max(1, height));
        swapDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapDesc.OutputWindow = hwnd;
        swapDesc.SampleDesc.Count = 1;
        swapDesc.SampleDesc.Quality = 0;
        swapDesc.Windowed = TRUE;
        swapDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
        result = factory->CreateSwapChain(device, &swapDesc, &swapChain);
    }

    if (selectedAdapter) 
        selectedAdapter->Release();

    if (factory)
        factory->Release();

    if (FAILED(result)) 
    {
        std::cerr << "Direct3D device/swapchain initialization failed: " << hresultToString(result) << "\n";
        releaseAllResources();
        return false;
    }

    return true;
}

bool Render::createRenderTargets() 
{
    ID3D11Texture2D* backBuffer = nullptr;
    HRESULT hr = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
    if (FAILED(hr))
    {
        std::cerr << "SwapChain GetBuffer failed: " << hresultToString(hr) << "\n";
        return false;
    }

    hr = device->CreateRenderTargetView(backBuffer, nullptr, &renderTargetView);
    safeRelease(backBuffer);
    if (FAILED(hr)) 
    {
        std::cerr << "CreateRenderTargetView failed: " << hresultToString(hr) << "\n";
        return false;
    }

    D3D11_TEXTURE2D_DESC depthDesc{};
    depthDesc.Width = static_cast<UINT>(std::max(1, width));
    depthDesc.Height = static_cast<UINT>(std::max(1, height));
    depthDesc.MipLevels = 1;
    depthDesc.ArraySize = 1;
    depthDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    depthDesc.SampleDesc.Count = 1;
    depthDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;

    hr = device->CreateTexture2D(&depthDesc, nullptr, &depthStencilTexture);
    if (FAILED(hr)) 
    {
        std::cerr << "Create depth texture failed: " << hresultToString(hr) << "\n";
        return false;
    }

    hr = device->CreateDepthStencilView(depthStencilTexture, nullptr, &depthStencilView);
    if (FAILED(hr))
    {
        std::cerr << "CreateDepthStencilView failed: " << hresultToString(hr) << "\n";
        return false;
    }

    D3D11_VIEWPORT vp{};
    vp.Width = static_cast<FLOAT>(std::max(1, width));
    vp.Height = static_cast<FLOAT>(std::max(1, height));
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0.0f;
    vp.TopLeftY = 0.0f;
    context->RSSetViewports(1, &vp);

    return true;
}

bool Render::compileShaderFromFile(const std::wstring& path, const char* target, ID3DBlob** outBlob) const
{
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_PACK_MATRIX_ROW_MAJOR;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    ID3DBlob* errors = nullptr;
    HRESULT hr = D3DCompileFromFile(path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", target, flags, 0, outBlob, &errors);
    if (FAILED(hr)) 
    {
        if (errors) 
        {
            OutputDebugStringA(static_cast<const char*>(errors->GetBufferPointer()));
            std::cerr << static_cast<const char*>(errors->GetBufferPointer()) << "\n";
            errors->Release();
        }
        std::wcerr << L"D3DCompileFromFile failed for " << path << L"\n";
        return false;
    }

    if (errors) 
        errors->Release();

    return true;
}

bool Render::createShadersAndInputLayout() 
{
    static const D3D11_INPUT_ELEMENT_DESC inputDesc[] =
    {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0}
    };

    ID3DBlob* LineVSCode = nullptr;
    if (!compileShaderFromFile(shaderPath(L"LineVertexShader.vs"), "vs_5_0", &LineVSCode))
        return false;

    HRESULT hr = device->CreateVertexShader(LineVSCode->GetBufferPointer(), LineVSCode->GetBufferSize(), nullptr, &solidVertexShader);
    if (FAILED(hr)) 
    {
        safeRelease(LineVSCode);
        std::cerr << "CreateVertexShader SolidVertex failed: " << hresultToString(hr) << "\n";
        return false;
    }

    hr = device->CreateInputLayout(inputDesc, 2, LineVSCode->GetBufferPointer(), LineVSCode->GetBufferSize(), &inputLayout);
    safeRelease(LineVSCode);
    if (FAILED(hr)) 
    {
        std::cerr << "CreateInputLayout failed: " << hresultToString(hr) << "\n";
        return false;
    }

    ID3DBlob* particleVSCode = nullptr;
    if (!compileShaderFromFile(shaderPath(L"ParticleVertex.vs"), "vs_5_0", &particleVSCode)) 
        return false;
    hr = device->CreateVertexShader(particleVSCode->GetBufferPointer(), particleVSCode->GetBufferSize(), nullptr, &particleVertexShader);
    safeRelease(particleVSCode);
    if (FAILED(hr)) 
    {
        std::cerr << "CreateVertexShader ParticleVertex failed: " << hresultToString(hr) << "\n";
        return false;
    }

    ID3DBlob* particleGSCode = nullptr;
    if (!compileShaderFromFile(shaderPath(L"ParticleGeometry.gs"), "gs_5_0", &particleGSCode))
        return false;

    hr = device->CreateGeometryShader(particleGSCode->GetBufferPointer(), particleGSCode->GetBufferSize(), nullptr, &particleGeometryShader);
    safeRelease(particleGSCode);
    if (FAILED(hr)) 
    {
        std::cerr << "CreateGeometryShader ParticleGeometry failed: " << hresultToString(hr) << "\n";
        return false;
    }

    ID3DBlob* pixelCode = nullptr;
    if (!compileShaderFromFile(shaderPath(L"ColorPixel.ps"), "ps_5_0", &pixelCode)) 
        return false;

    hr = device->CreatePixelShader(pixelCode->GetBufferPointer(), pixelCode->GetBufferSize(), nullptr, &pixelShader);
    safeRelease(pixelCode);
    if (FAILED(hr)) 
    {
        std::cerr << "CreatePixelShader ColorPixel failed: " << hresultToString(hr) << "\n";
        return false;
    }

    return true;
}

void Render::handleResize(HWND hwnd) 
{
    if (!swapChain || !context || !hwnd)
    {
        return;
    }

    RECT rc{};
    GetClientRect(hwnd, &rc);
    width = std::max<LONG>(1, rc.right - rc.left);
    height = std::max<LONG>(1, rc.bottom - rc.top);

    context->OMSetRenderTargets(0, nullptr, nullptr);
    releaseRenderTargets();

    HRESULT hr = swapChain->ResizeBuffers(2, static_cast<UINT>(width), static_cast<UINT>(height), DXGI_FORMAT_R8G8B8A8_UNORM, 0);
    if (FAILED(hr)) 
    {
        MessageBoxW(hwnd, L"ResizeBuffers failed", L"DirectX error", MB_OK | MB_ICONERROR);
        return;
    }

    if (!createRenderTargets()) 
    {
        MessageBoxW(hwnd, L"Back buffer reconfiguration failed", L"DirectX error", MB_OK | MB_ICONERROR);
    }
}

bool Render::createRasterizerDepthBlendStates() 
{
    D3D11_RASTERIZER_DESC rasterDesc{};
    rasterDesc.FillMode = D3D11_FILL_SOLID;
    rasterDesc.CullMode = D3D11_CULL_NONE;
    rasterDesc.DepthClipEnable = TRUE;

    HRESULT hr = device->CreateRasterizerState(&rasterDesc, &rasterizerState);
    if (FAILED(hr)) 
    {
        std::cerr << "CreateRasterizerState failed: " << hresultToString(hr) << "\n";
        return false;
    }

    D3D11_DEPTH_STENCIL_DESC depthDesc{};
    depthDesc.DepthEnable = TRUE;
    depthDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    depthDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;

    hr = device->CreateDepthStencilState(&depthDesc, &depthState);
    if (FAILED(hr)) 
    {
        std::cerr << "CreateDepthStencilState failed: " << hresultToString(hr) << "\n";
        return false;
    }

    D3D11_BLEND_DESC blendDesc{};
    blendDesc.RenderTarget[0].BlendEnable = TRUE;
    blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    hr = device->CreateBlendState(&blendDesc, &blendState);
    if (FAILED(hr)) 
    {
        std::cerr << "CreateBlendState failed: " << hresultToString(hr) << "\n";
        return false;
    }

    return true;
}

bool Render::createConstantBuffer() 
{
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = sizeof(SceneConstantBuffer);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

    HRESULT hr = device->CreateBuffer(&desc, nullptr, &constantBuffer);
    if (FAILED(hr)) 
    {
        std::cerr << "Create constant buffer failed: " << hresultToString(hr) << "\n";
        return false;
    }
    return true;
}

bool Render::isKeyPressed(int virtualKey) const 
{
    if (!hwnd) 
        return false;

    return (GetAsyncKeyState(virtualKey) & 0x8000) != 0;
}

void Render::setWindowTitle(const std::string& title)
{
    if (hwnd) 
    {
        std::wstring wtitle(title.begin(), title.end());
        SetWindowTextW(hwnd, wtitle.c_str());
    }
}

void Render::renderFrame(const CollisionScene& scene, const std::vector<ParticleState>& particles, float particleRadius)
{
    if (!device || !context) 
    {
        return;
    }

    if (width <= 0 || height <= 0) 
    {
        return;
    }

    D3D11_VIEWPORT viewport{};
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
    viewport.Width = static_cast<float>(width);
    viewport.Height = static_cast<float>(height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;

    float clearColor[4] = { 0.035f, 0.04f, 0.052f, 1.0f };
    context->OMSetRenderTargets(1, &renderTargetView, depthStencilView);
    context->RSSetViewports(1, &viewport);
    context->ClearRenderTargetView(renderTargetView, clearColor);
    context->ClearDepthStencilView(depthStencilView, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

    XMVECTOR eye = XMVectorSet(11.5f, 3.5f, -28.0f, 1.0f);
    XMVECTOR at = XMVectorSet(0.0f, -3.0f, 0.0f, 1.0f);
    XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
    XMMATRIX view = XMMatrixLookAtLH(eye, at, up);
    XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(50.0f),
        static_cast<float>(width) / static_cast<float>(height),
        0.1f, 100.0f);

    SceneConstantBuffer cb{};
    cb.mvp = view * proj;
    cb.invViewport = XMFLOAT2(2.0f / static_cast<float>(width), 2.0f / static_cast<float>(height));
    cb.pointSize = std::max(3.0f, particleRadius * 55.0f);

    context->UpdateSubresource(constantBuffer, 0, nullptr, &cb, 0, 0);
    context->VSSetConstantBuffers(0, 1, &constantBuffer);
    context->GSSetConstantBuffers(0, 1, &constantBuffer);

    context->IASetInputLayout(inputLayout);
    context->RSSetState(rasterizerState);
    context->OMSetDepthStencilState(depthState, 0);
    float blendFactor[4] = { 0, 0, 0, 0 };
    context->OMSetBlendState(blendState, blendFactor, 0xffffffff);

    drawSceneObjects(scene);
    drawParticles(particles, particleRadius);

    HRESULT hr = swapChain->Present(1, 0);
    if (FAILED(hr)) 
    {
        std::cerr << "SwapChain Present failed: " << hresultToString(hr) << "\n";
    }
}

bool Render::ensureVertexBufferCapacity(ID3D11Buffer** buffer, size_t& capacity, size_t requiredBytes) 
{
    if (requiredBytes == 0) 
    {
        return true;
    }

    if (*buffer && capacity >= requiredBytes)
    {
        return true;
    }

    safeRelease(*buffer);
    capacity = std::max(requiredBytes, capacity * 2 + sizeof(Vertex) * 1024);

    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(capacity);
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    HRESULT hr = device->CreateBuffer(&desc, nullptr, buffer);
    if (FAILED(hr)) 
    {
        std::cerr << "Create dynamic vertex buffer failed: " << hresultToString(hr) << "\n";
        return false;
    }
    return true;
}

bool Render::updateVertexBufferData(ID3D11Buffer* buffer, const void* data, size_t bytes) 
{
    if (bytes == 0) 
    {
        return true;
    }

    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = context->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    if (FAILED(hr)) 
    {
        std::cerr << "Map vertex buffer failed: " << hresultToString(hr) << "\n";
        return false;
    }

    std::memcpy(mapped.pData, data, bytes);
    context->Unmap(buffer, 0);
    return true;
}

void Render::addBoxWireframe(const Box& box, float r, float g, float b, std::vector<Vertex>& lines) 
{
    float3 c = box.center;
    float3 h = box.halfSize;
    float3 p[8] = {
        make_float3(c.x - h.x, c.y - h.y, c.z - h.z),
        make_float3(c.x + h.x, c.y - h.y, c.z - h.z),
        make_float3(c.x + h.x, c.y + h.y, c.z - h.z),
        make_float3(c.x - h.x, c.y + h.y, c.z - h.z),
        make_float3(c.x - h.x, c.y - h.y, c.z + h.z),
        make_float3(c.x + h.x, c.y - h.y, c.z + h.z),
        make_float3(c.x + h.x, c.y + h.y, c.z + h.z),
        make_float3(c.x - h.x, c.y + h.y, c.z + h.z)
    };
    int edges[12][2] = {
        {0,1}, {1,2}, {2,3}, {3,0},
        {4,5}, {5,6}, {6,7}, {7,4},
        {0,4}, {1,5}, {2,6}, {3,7}
    };
    for (auto& edge : edges) 
    {
        addLine(lines, p[edge[0]], p[edge[1]], r, g, b);
    }
}

void Render::addSphereWireframe(const Sphere& sphere, std::vector<Vertex>& lines) 
{
    for (int ring = 0; ring < kSphereRings; ++ring) 
    {
        float angle = ring * kPi / kSphereRings;
        float cosA = cosf(angle);
        float sinA = sinf(angle);

        for (int i = 0; i < kSphereSegments; ++i) 
        {
            float a0 = 2.0f * kPi * static_cast<float>(i) / kSphereSegments;
            float a1 = 2.0f * kPi * static_cast<float>(i + 1) / kSphereSegments;

            float x0 = cosf(a0) * sphere.radius;
            float y0 = sinf(a0) * sphere.radius;
            float z0 = 0.0f;
            float rx0 = x0 * cosA + z0 * sinA;
            float ry0 = y0;
            float rz0 = -x0 * sinA + z0 * cosA;

            float x1 = cosf(a1) * sphere.radius;
            float y1 = sinf(a1) * sphere.radius;
            float z1 = 0.0f;
            float rx1 = x1 * cosA + z1 * sinA;
            float ry1 = y1;
            float rz1 = -x1 * sinA + z1 * cosA;

            addLine(lines,
                make_float3(sphere.center.x + rx0, sphere.center.y + ry0, sphere.center.z + rz0),
                make_float3(sphere.center.x + rx1, sphere.center.y + ry1, sphere.center.z + rz1),
                kSphereColor[0], kSphereColor[1], kSphereColor[2]);
        }
    }
}

void Render::addPlaneWireframe(const Plane& plane, std::vector<Vertex>& lines) 
{
    float size = plane.halfSize;
    float3 n = normalize3(plane.normal);
    float3 helper = std::fabs(n.y) < 0.9f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 u = normalize3(cross3(helper, n));
    float3 v = normalize3(cross3(n, u));

    float3 p0 = plane.point + u * (-size) + v * (-size);
    float3 p1 = plane.point + u * (size)+v * (-size);
    float3 p2 = plane.point + u * (size)+v * (size);
    float3 p3 = plane.point + u * (-size) + v * (size);

    addLine(lines, p0, p1, kPlaneBorderColor[0], kPlaneBorderColor[1], kPlaneBorderColor[2]);
    addLine(lines, p1, p2, kPlaneBorderColor[0], kPlaneBorderColor[1], kPlaneBorderColor[2]);
    addLine(lines, p2, p3, kPlaneBorderColor[0], kPlaneBorderColor[1], kPlaneBorderColor[2]);
    addLine(lines, p3, p0, kPlaneBorderColor[0], kPlaneBorderColor[1], kPlaneBorderColor[2]);

    constexpr int gridLines = 8;
    for (int i = -gridLines; i <= gridLines; ++i) 
    {
        float t = size * static_cast<float>(i) / gridLines;
        addLine(lines, plane.point + u * t + v * (-size),
            plane.point + u * t + v * (size), kPlaneGridColor[0], kPlaneGridColor[1], kPlaneGridColor[2]);
        addLine(lines, plane.point + u * (-size) + v * t,
            plane.point + u * (size)+v * t, kPlaneGridColor[0], kPlaneGridColor[1], kPlaneGridColor[2]);
    }
}

void Render::addFloorGridLines(float minX, float maxX, float minZ, float maxZ, float y, int cells, std::vector<Vertex>& lines) 
{
    for (int i = -cells; i <= cells; ++i) 
    {
        float coord = static_cast<float>(i);
        addLine(lines, make_float3(coord, y, minZ), make_float3(coord, y, maxZ),
            kFloorMajorColor[0], kFloorMajorColor[1], kFloorMajorColor[2]);
        addLine(lines, make_float3(minX, y, coord), make_float3(maxX, y, coord),
            kFloorMajorColor[0], kFloorMajorColor[1], kFloorMajorColor[2]);
    }
}

void Render::buildSceneWireframeGeometry(const CollisionScene& scene, std::vector<Vertex>& lines) 
{
    addFloorGridLines(-kFloorExtent, kFloorExtent, -kFloorExtent, kFloorExtent, kFloorY, kFloorCells, lines);

    addSphereWireframe(scene.sphere, lines);

    for (int i = 0; i < scene.planeCount && i < MAX_PLANES; ++i) 
    {
        addPlaneWireframe(scene.planes[i], lines);
    }

    Box basketBox{};
    basketBox.center = make_float3(
        (scene.basket.minCorner.x + scene.basket.maxCorner.x) * 0.5f,
        (scene.basket.minCorner.y + scene.basket.maxCorner.y) * 0.5f,
        (scene.basket.minCorner.z + scene.basket.maxCorner.z) * 0.5f
    );

    basketBox.halfSize = make_float3(
        (scene.basket.maxCorner.x - scene.basket.minCorner.x) * 0.5f,
        (scene.basket.maxCorner.y - scene.basket.minCorner.y) * 0.5f,
        (scene.basket.maxCorner.z - scene.basket.minCorner.z) * 0.5f
    );

    addBoxWireframe(basketBox, kBasketColor[0], kBasketColor[1], kBasketColor[2], lines);
}

void Render::drawSceneObjects(const CollisionScene& scene) 
{
    std::vector<Vertex> lines;
    lines.reserve(4096);

    buildSceneWireframeGeometry(scene, lines);

    if (lines.empty()) 
    {
        return;
    }

    size_t bytes = lines.size() * sizeof(Vertex);
    if (!ensureVertexBufferCapacity(&lineVertexBuffer, lineVertexCapacity, bytes)) 
        return;
    if (!updateVertexBufferData(lineVertexBuffer, lines.data(), bytes)) 
        return;

    UINT stride = sizeof(Vertex);
    UINT offset = 0;
    context->IASetVertexBuffers(0, 1, &lineVertexBuffer, &stride, &offset);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
    context->VSSetShader(solidVertexShader, nullptr, 0);
    context->GSSetShader(nullptr, nullptr, 0);
    context->PSSetShader(pixelShader, nullptr, 0);
    context->Draw(static_cast<UINT>(lines.size()), 0);
}

void Render::drawParticles(const std::vector<ParticleState>& particles, float /*particleRadius*/) 
{
    if (particles.empty()) 
    {
        return;
    }

    std::vector<Vertex> vertices;
    vertices.reserve(particles.size());
    for (const auto& particle : particles) 
    {
        if (particle.type == BOUNCE) 
        {
            vertices.push_back(makeVertex(particle.position, 0.20f, 0.75f, 0.90f, 0.85f));
        }
        else 
        {
            vertices.push_back(makeVertex(particle.position, 0.95f, 0.75f, 0.65f, 0.85f));
        }
    }

    size_t bytes = vertices.size() * sizeof(Vertex);
    if (!ensureVertexBufferCapacity(&particleVertexBuffer, particleVertexCapacity, bytes)) 
        return;
    if (!updateVertexBufferData(particleVertexBuffer, vertices.data(), bytes)) 
        return;

    UINT stride = sizeof(Vertex);
    UINT offset = 0;
    context->IASetVertexBuffers(0, 1, &particleVertexBuffer, &stride, &offset);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
    context->VSSetShader(particleVertexShader, nullptr, 0);
    context->GSSetShader(particleGeometryShader, nullptr, 0);
    context->PSSetShader(pixelShader, nullptr, 0);
    context->Draw(static_cast<UINT>(vertices.size()), 0);
    context->GSSetShader(nullptr, nullptr, 0);
}

void Render::releaseRenderTargets() 
{
    safeRelease(renderTargetView);
    safeRelease(depthStencilView);
    safeRelease(depthStencilTexture);
}

void Render::shutdownRender() 
{
    releaseAllResources();
}

void Render::releaseAllResources() 
{
    if (context) 
    {
        context->OMSetRenderTargets(0, nullptr, nullptr);
        context->ClearState();
        context->Flush();
    }

    safeRelease(annotation);

    safeRelease(particleVertexBuffer);
    safeRelease(lineVertexBuffer);
    safeRelease(constantBuffer);
    safeRelease(blendState);
    safeRelease(depthState);
    safeRelease(rasterizerState);
    safeRelease(pixelShader);
    safeRelease(particleGeometryShader);
    safeRelease(particleVertexShader);
    safeRelease(solidVertexShader);
    safeRelease(inputLayout);
    releaseRenderTargets();
    safeRelease(swapChain);
    safeRelease(context);
    safeRelease(device);

    hwnd = nullptr;
}