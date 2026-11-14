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
    bool p_socket_initialized = false;

    CCRenderTexture* texture = nullptr;
    std::vector<uint8_t> gray;
    const int scaling_factor = 4;

    struct FrameHeader
    {
        uint32_t dead;
        uint32_t level_won;
        uint32_t quit;
        uint32_t width;
        uint32_t height;
        uint32_t bytes;
    };

    static_assert(sizeof(FrameHeader) == 24);

	struct SimulationState 
	{
        bool jump_requested = false;
		bool jump_held = false;
		bool step_requested = false;
		bool injecting_input = false;
        bool frame_requested = false;
        bool terminal_requested = false;
        bool reset_pending = false;
    };

    SimulationState sim;

    bool dead = false;
    bool won = false;
    bool quit = false;

    constexpr float step_dt = 1.0f / 240.0f;
}

void reset_sim()
{
    sim.jump_requested = false;
    sim.jump_held = false;
    sim.step_requested = false;
    sim.injecting_input = false;
    sim.frame_requested = true;
    sim.reset_pending = false;
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
        if (p_socket_initialized) {return false;}

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
            log::info("Python socket connected!");
            p_socket_initialized = true;

            DWORD timeout_ms = 2000;
            if (setsockopt(python_socket, SOL_SOCKET, SO_SNDTIMEO,
               reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms)) == SOCKET_ERROR)
            {
                log::error("Failed to set send timeout: {}", WSAGetLastError());
                closesocket(python_socket);
                python_socket = INVALID_SOCKET;
                return false;
            }
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

    texture = CCRenderTexture::create(static_cast<int>(size.width / scaling_factor), static_cast<int>(size.height / scaling_factor));

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

    if (sim.terminal_requested)
    {
        if (!(dead || won || quit)) 
        {
            log::info("Terminal requested but no terminal value set to true");
            return;
        }

        FrameHeader header{htonl(dead), htonl(won), htonl(quit), 
                            htonl(0), htonl(0), htonl(0)};

        if (!change_blocking_mode(python_socket, 0) ||
            !send_data(&header, sizeof(header))) 
        {
            return;
        }

        change_blocking_mode(python_socket, 1);

        dead = false;
        won = false;
        quit = false;
        sim.terminal_requested = false;
        return;
    }
    else if (sim.reset_pending) {return;}

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

    kmGLScalef(1.0f / scaling_factor, 1.0f / scaling_factor, 1.0f);

    scene->visit();

    kmGLPopMatrix();

    texture->end();

    CCImage* image = texture->newCCImage(true);

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

    FrameHeader header{htonl(0), htonl(0), htonl(0), 
                        htonl(width), htonl(height), htonl(bytes)};

    gray.resize(width * height);
    for (int i = 0; i < width * height; i++)
    {
        int index = i * 4;

        gray[i] = static_cast<uint8_t>(
            0.299f * pixels[index] +
            0.587f * pixels[index + 1] +
            0.114f * pixels[index + 2]
        );
    }

    if (!change_blocking_mode(python_socket, 0) ||
        !send_data(&header, sizeof(header)) || 
        !send_data(gray.data(), bytes)) 
    {delete image; return;}

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

        //static auto last_time = std::chrono::steady_clock::now();
        //static int total_updates = 0;
        //static int successful_updates = 0;
        //total_updates++;
        //auto now = std::chrono::steady_clock::now();
        //double elapsed = std::chrono::duration<double>(now - last_time).count();
        //if (elapsed >= 5.0)
        //{
        //    log::info("Updates/sec: {:.1f}", total_updates / elapsed);
        //    log::info("Successful steps/sec: {:.1f}", successful_updates / elapsed);
        //    total_updates = 0;
        //    successful_updates = 0;
        //    last_time = now;
        //}

        if (sim.frame_requested || sim.terminal_requested) 
        {
            capture_and_send_data();
        }

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

        //successful_updates++;

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

        // Set up socket
        initialize_socket();

        // initialize a texture to load new frames into
        initialize_texture();

        reset_sim();

        sim.injecting_input = true;
        GJBaseGameLayer::handleButton(false, 1, true);
        sim.injecting_input = false;

        return true;
    }

    void resetLevel()
    {
        PlayLayer::resetLevel();
        log::info("resetLevel hook called");

        sim.injecting_input = true;
        GJBaseGameLayer::handleButton(false, 1, true);
        sim.injecting_input = false;

        reset_sim();
    }
    
    void destroyPlayer(PlayerObject* player, GameObject* cause) {
        bool wasDead = m_playerDied;

        PlayLayer::destroyPlayer(player, cause);

        // Only report a death if the player actually died
        if (!wasDead && m_playerDied) {
            dead = true;
            sim.terminal_requested = true;
            sim.reset_pending = true;

            log::info("Player actually died!");
        }
    }

    void levelComplete() {
        log::info("Level complete hook called!");

        won = true;
        sim.terminal_requested = true;
        sim.reset_pending = true;
        resetLevel(); // my reset level hook
    }

    void onQuit()
    {
        if (python_socket != INVALID_SOCKET)
        {
            quit = true;
            sim.terminal_requested = true;
            capture_and_send_data();
        }

        PlayLayer::onQuit();
    }
};