#include <chrono>

#include <winsock2.h>
#include <ws2tcpip.h>

#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

using namespace geode::prelude;

namespace 
{
    SOCKET sock = INVALID_SOCKET;
    SOCKET python_socket = INVALID_SOCKET;
    bool socket_initialized = false;

    CCRenderTexture* texture = nullptr;
    std::vector<uint8_t> gray;

    struct FrameHeader
    {
        uint32_t dead;
        uint32_t width;
        uint32_t height;
        uint32_t bytes;
    };

    static_assert(sizeof(FrameHeader) == 16);

	struct SimulationState 
	{
        bool jump_requested = false;
		bool jump_held = false;
		bool step_requested = false;
		bool injecting_input = false;
        bool frame_requested = false;
    };

    SimulationState sim;

    constexpr float step_dt = 1.0f / 240.0f;
}

void reset_sim()
{
    sim.jump_requested = false;
    sim.jump_held = false;
    sim.step_requested = false;
    sim.injecting_input = false;
    sim.frame_requested = true;
}

bool change_blocking_mode(SOCKET &socket, u_long mode)
{ // returns false to abort outer function
    if (ioctlsocket(socket, FIONBIO, &mode) == SOCKET_ERROR)
        {
            if (mode == 0) {log::error("Failed to enable blocking mode!");}
            else {log::error("Failed to enable nonblocking mode!");}
            closesocket(socket);
            socket = INVALID_SOCKET; 
            return false;
        }
    return true;
}

void initialize_socket()
{
    if (socket_initialized) {return;}

    WSADATA wsa;

    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
    {
        log::error("WSAStartup failed!");
        return;
    }

    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    if (sock == INVALID_SOCKET)
    {
        log::error("Failed to create socket!");
        WSACleanup();
        return;
    }

    sockaddr_in address{};

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(5555);

    if (bind(sock, reinterpret_cast<sockaddr*>(&address), sizeof(address)) 
        == SOCKET_ERROR)
    {
        log::error("Failed to bind socket!");
        closesocket(sock);
        sock = INVALID_SOCKET;
        WSACleanup();
        return;
    }

    if (listen(sock, 1) == SOCKET_ERROR)
    {
        log::error("Failed to listen: {}", WSAGetLastError());
        closesocket(sock);
        sock = INVALID_SOCKET;
        WSACleanup();
        return;
    }

    if (!change_blocking_mode(sock, 1))
    {
        WSACleanup();
        return;
    }

    socket_initialized = true;

    log::info("Socket listening on port 5555!");
}

bool check_socket()
{ // return false to abort outer function
    if (!socket_initialized) {return false;}

    if (python_socket == INVALID_SOCKET) 
    {
        python_socket = accept(sock, nullptr, nullptr);

        if (python_socket == INVALID_SOCKET)
        {
            int error = WSAGetLastError();

            if (error != WSAEWOULDBLOCK)
            {
                log::error("Accept failed: {}", error);
            }

            return false;
        }
        else 
        {
            DWORD timeout_ms = 2000;
            if (setsockopt(python_socket, SOL_SOCKET, SO_SNDTIMEO,
               reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms)) == SOCKET_ERROR)
            {
                log::error("Failed to set send timeout: {}", WSAGetLastError());
                closesocket(python_socket);
                python_socket = INVALID_SOCKET;
                return false;
            }

            log::info("Python connected!");
            sim.frame_requested = true;
        }
    }

    return true;
}

void initialize_texture()
{
    CCDirector* director = CCDirector::sharedDirector();
    CCScene* scene = director->getRunningScene();

    if (!scene) 
    {
        log::error("Failed to fetch Running Scene");
        return;
    }

    auto size = director->getWinSize();

    texture = CCRenderTexture::create(static_cast<int>(size.width / 4), static_cast<int>(size.height / 4));

    if (!texture) 
    {
        log::error("Failed to create RenderTexture");
        return;
    }

    texture->retain();
}

bool send_data(const void *data, int bytes)
{ // returns false to abort outer function
    int sent;
    int total = 0;
    while (total < bytes)
    {
        sent = send(python_socket, reinterpret_cast<const char*>(data) + total, bytes - total, 0);

        if (sent <= 0) 
        {
            if (sent == 0) {log::error("Send failed: Python disconnected");}
            else {log::error("Send failed: {}", WSAGetLastError());}
            closesocket(python_socket);
            python_socket = INVALID_SOCKET; 
            return false;
        }
        else {total += sent;}
    }
    return true;
}

void receive_python_input()
{
    if (!check_socket()) {return;}

    char message;

    int received = recv(python_socket, &message, 1, 0);

    if (received == 1)
    {
        if (message == '1')
        {
            sim.jump_requested = true;
            sim.step_requested = true;
        }
        else if (message == '0')
        {
            sim.jump_requested = false;
            sim.step_requested = true;
        }
        else
        {
            log::error("Recived invalid action: {}", message);
        }
    }
    else if (received == SOCKET_ERROR)
    {
        int error = WSAGetLastError();

        if (error != WSAEWOULDBLOCK)
        {
            log::error("Socket receive failed: {}", error);
            closesocket(python_socket);
            python_socket = INVALID_SOCKET;
        }
    }
    else if (received == 0)
    {
        log::info("Python disconnected!");

        closesocket(python_socket);
        python_socket = INVALID_SOCKET;
    }

    //log::info("Python sent: {}", message);
}

void capture_and_send_data()
{
    if (!check_socket()) {return;}

    static auto last_time = std::chrono::steady_clock::now();

    static double t_image = 0;
    static double t_send_data = 0;
    static double t_grey = 0;
    static int frames = 0;

    if (PlayLayer::get()->m_playerDied)
    {
        FrameHeader header{htonl(1), htonl(0), htonl(0), htonl(0)};

        if (!change_blocking_mode(python_socket, 0) ||
            !send_data(&header, sizeof(header))) 
        {
            return;
        }

        change_blocking_mode(python_socket, 1);

        sim.frame_requested = false;
        return;
    }

    CCDirector* director = CCDirector::sharedDirector();
    CCScene* scene = director->getRunningScene();

    if (!scene) 
    {
        log::error("Failed to fetch Running Scene");
        return;
    }
    if (!texture) 
    {
        log::error("No texture");
        return;
    }

    texture->beginWithClear(0, 0, 0, 1);

    kmGLMatrixMode(KM_GL_MODELVIEW);
    kmGLPushMatrix();

    kmGLScalef(0.25f, 0.25f, 1.0f);

    scene->visit();

    kmGLPopMatrix();

    texture->end();

    auto start = std::chrono::steady_clock::now();
    CCImage* image = texture->newCCImage(true);
    t_image += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start
    ).count();

    if (!image) 
    {
        log::error("Failed to fetch image");
        return;
    }

    auto* pixels = image->getData();

    int width = image->getWidth();
    int height = image->getHeight();
    int bytes = width * height;

    //log::info("Captured {}x{} frame, {} bytes", width, height, bytes);

    FrameHeader header{htonl(0), htonl(width), htonl(height), htonl(bytes)};

    gray.resize(width * height);
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < width * height; i++)
    {
        int index = i * 4;

        gray[i] = static_cast<uint8_t>(
            0.299f * pixels[index] +
            0.587f * pixels[index + 1] +
            0.114f * pixels[index + 2]
        );
    }
    t_grey += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start
    ).count();

    if (!change_blocking_mode(python_socket, 0) ||
        !send_data(&header, sizeof(header))) 
    {delete image; return;}

    start = std::chrono::steady_clock::now();
    if (!send_data(gray.data(), bytes)) {delete image; return;}
    t_send_data += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start
    ).count();

    frames++;
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - last_time).count();
    if (elapsed >= 1.0)
    {
        log::info("Gray: {:.2f} ms/frame", t_grey / frames);
        log::info("CCImage: {:.2f} ms/frame", t_image / frames);
        log::info("Send: {:.2f} ms/frame", t_send_data / frames);   

        t_image = 0.0;
        t_send_data = 0.0;
        t_grey = 0.0;
        frames = 0;
        last_time = now;
    }

    change_blocking_mode(python_socket, 1);

    delete image;
    sim.frame_requested = false;
}

class $modify(MyGameLayer, GJBaseGameLayer) 
{
    void update(float dt) 
	{
        // Allow non-gameplay layers to update normally
        if (!PlayLayer::get())
        {
            GJBaseGameLayer::update(dt);
            return;
        }

        static auto last_time = std::chrono::steady_clock::now();

        static int total_updates = 0;
        static int successful_updates = 0;

        total_updates++;

        auto now = std::chrono::steady_clock::now();

        double elapsed = std::chrono::duration<double>(now - last_time).count();

        if (elapsed >= 1.0)
        {
            log::info("Updates/sec: {:.1f}", total_updates / elapsed);
            log::info("Successful steps/sec: {:.1f}", successful_updates / elapsed);

            total_updates = 0;
            successful_updates = 0;
            last_time = now;
        }

        if (sim.frame_requested) {capture_and_send_data();}

        // Check whether Python sent a command
        receive_python_input();

        // Wait for Python to request a step
        if (!sim.step_requested) {return;}

		sim.step_requested = false;

        // Only send input when the button state changes
        if (sim.jump_requested != sim.jump_held)
		{
            sim.injecting_input = true;
            handleButton(sim.jump_requested, 1, true);
            sim.injecting_input = false;

            sim.jump_held = sim.jump_requested;
    	}

        // Advance the simulation
        GJBaseGameLayer::update(step_dt);

        successful_updates++;

        // request another frame
        sim.frame_requested = true;
        capture_and_send_data();
    }

	void handleButton(bool down, int button, bool player1) 
    {
        // Ignore real gameplay inputs during simulation
        if (!sim.injecting_input) {return;}

        // Otherwise process the input normally
        GJBaseGameLayer::handleButton(down, button, player1);
    }
};

class $modify(MyPlayLayer, PlayLayer)
{
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects)
    {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) 
		{
            return false;
        }

        // Set up socket if necessary
        initialize_socket();

        // initialize a texture to load new frames into
        initialize_texture();

        reset_sim();

        return true;
    }

    void resetLevel()
    {
        PlayLayer::resetLevel();
        reset_sim();
    }

    void onQuit()
    {
        if (texture)
        {
            texture->release();
            texture = nullptr;
        }

        if (python_socket != INVALID_SOCKET)
        {
            closesocket(python_socket);
            python_socket = INVALID_SOCKET;
        }

        if (sock != INVALID_SOCKET)
        {
            closesocket(sock);
            sock = INVALID_SOCKET;
        }

        if (socket_initialized) {WSACleanup();}

        socket_initialized = false;

        PlayLayer::onQuit();
    }
};