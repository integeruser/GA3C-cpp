#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

#include "gym-uds.h"
#include "gym-uds.pb.h"


namespace gym_uds
{
template<typename T>
T Environment::recv_message()
{
    char buf[1024];

    auto nread = read(sock, buf, 4);
    const int message_pb_len = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];

    nread = read(sock, buf, message_pb_len);
    std::string message_pb(buf, nread);

    T message;
    if (!message.ParseFromString(message_pb)) {
        std::cerr << "Failed to parse message." << std::endl;
        std::exit(1);
    }
    return message;
}

template<typename T>
void Environment::send_message(const T& message)
{
    std::string message_pb;
    if (!message.SerializeToString(&message_pb)) {
        std::cerr << "Failed to write message." << std::endl;
        std::exit(1);
    }

    const int message_pb_len = message_pb.size();
    auto nwrite = write(sock, (char *)&message_pb_len, 4);
    nwrite = write(sock, message_pb.c_str(), message_pb.size());
}


Environment::Environment(const std::string& sock_filepath)
{
    struct sockaddr_un server_addr = {};
    #ifdef __APPLE__
    server_addr.sun_len = sizeof server_addr;
    #endif
    server_addr.sun_family = AF_UNIX;
    std::strncpy(server_addr.sun_path, sock_filepath.c_str(), sizeof server_addr.sun_path);

    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) { std::perror("gym::Environment::socket"); std::exit(1); }

    #ifdef __APPLE__
    const auto addr_len = SUN_LEN(&server_addr);
    #else
    const auto addr_len = std::strlen(server_addr.sun_path) + sizeof server_addr.sun_family;
    #endif
    const auto conn = connect(sock, (struct sockaddr *)&server_addr, addr_len);
    if (conn < 0) { std::perror("gym::Environment::connect"); std::exit(1); }
}


observation_t Environment::reset()
{
    Request request;
    request.set_type(Request::RESET);
    send_message<Request>(request);

    State state = recv_message<State>();
    observation_t observation;
    std::copy_n(state.observation().data().cbegin(), state.observation().data().size(), std::back_inserter(observation));
    return observation;
}

state_t Environment::step(const action_t& action_value)
{
    Request request;
    request.set_type(Request::STEP);
    send_message<Request>(request);

    Action action;
    action.set_value(action_value);
    send_message<Action>(action);

    State state = recv_message<State>();
    observation_t observation;
    std::copy_n(state.observation().data().cbegin(), state.observation().data().size(), std::back_inserter(observation));
    return {observation, state.reward(), state.done()};
}


action_t Environment::sample()
{
    Request request;
    request.set_type(Request::SAMPLE);
    send_message<Request>(request);

    Action action = recv_message<Action>();
    return action.value();
}
}
