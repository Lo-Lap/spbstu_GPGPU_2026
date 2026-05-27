#ifndef RENDER
#define RENDER

#include <windows.h>
#include <d3dcommon.h>

#include <string>
#include <vector>

#include <DirectXMath.h>

#include <cuda_runtime.h>

#include "SceneTypes.h"
#include "ParticleEngine.cuh"

enum KeyCode 
{
    KEY_ESCAPE = VK_ESCAPE,
    KEY_1 = '1',
    KEY_2 = '2',
    KEY_3 = '3',
    KEY_P = 'P'
};

struct Vertex
{
    float x, y, z;
    float r, g, b, a;
};

struct ID3D11Device;
struct ID3D11DeviceContext;
struct IDXGISwapChain;
struct ID3D11RenderTargetView;
struct ID3D11Texture2D;
struct ID3D11DepthStencilView;
struct ID3D11InputLayout;
struct ID3D11VertexShader;
struct ID3D11GeometryShader;
struct ID3D11PixelShader;
struct ID3D11Buffer;
struct ID3D11RasterizerState;
struct ID3D11DepthStencilState;
struct ID3D11BlendState;
struct ID3DUserDefinedAnnotation;

class Render 
{
public:
    Render() = default;
    ~Render();

    Render(const Render&) = delete;
    Render& operator=(const Render&) = delete;

    bool initializeRender(HWND hwnd, int width, int height);
    void shutdownRender();
    void handleResize(HWND hwnd);

    bool isKeyPressed(int virtualKey) const;
    void setWindowTitle(const std::string& title);
    HWND getWindowHandle() const { return hwnd; }

    void renderFrame(const CollisionScene& scene, const std::vector<ParticleState>& particles, float particleRadius );

private:
    HWND hwnd = nullptr;
    int width = 1280;
    int height = 720;

    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGISwapChain* swapChain = nullptr;
    ID3D11RenderTargetView* renderTargetView = nullptr;
    ID3D11Texture2D* depthStencilTexture = nullptr;
    ID3D11DepthStencilView* depthStencilView = nullptr;

    ID3D11InputLayout* inputLayout = nullptr;
    ID3D11VertexShader* solidVertexShader = nullptr;
    ID3D11VertexShader* particleVertexShader = nullptr;
    ID3D11GeometryShader* particleGeometryShader = nullptr;
    ID3D11PixelShader* pixelShader = nullptr;

    ID3D11Buffer* constantBuffer = nullptr;
    ID3D11Buffer* particleVertexBuffer = nullptr;
    ID3D11Buffer* lineVertexBuffer = nullptr;
    size_t particleVertexCapacity = 0;
    size_t lineVertexCapacity = 0;

    ID3D11RasterizerState* rasterizerState = nullptr;
    ID3D11DepthStencilState* depthState = nullptr;
    ID3D11BlendState* blendState = nullptr;
    ID3DUserDefinedAnnotation* annotation = nullptr;

    bool createDeviceAndSwapChain();
    bool createRenderTargets();
    bool createShadersAndInputLayout();
    bool createRasterizerDepthBlendStates();
    bool createConstantBuffer();
    void releaseRenderTargets();
    void releaseAllResources();

    bool compileShaderFromFile(const std::wstring& path, const char* target, ID3DBlob** outBlob) const;

    void drawParticles(const std::vector<ParticleState>& particles, float particleRadius);
    void drawSceneObjects(const CollisionScene& scene);

    void addBoxWireframe(const Box& box, float r, float g, float b, std::vector<Vertex>& lines);
    void addSphereWireframe(const Sphere& sphere, std::vector<Vertex>& lines);
    void addPlaneWireframe(const Plane& plane, std::vector<Vertex>& lines);
    void addFloorGridLines(float minX, float maxX, float minZ, float maxZ, float y, int cells, std::vector<Vertex>& lines);
    void buildSceneWireframeGeometry(const CollisionScene& scene, std::vector<Vertex>& lines);

    bool ensureVertexBufferCapacity(ID3D11Buffer** buffer, size_t& capacity, size_t requiredBytes);
    bool updateVertexBufferData(ID3D11Buffer* buffer, const void* data, size_t bytes);
};

#endif