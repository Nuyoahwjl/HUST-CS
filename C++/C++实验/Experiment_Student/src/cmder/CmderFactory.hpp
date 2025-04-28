#pragma once

#include <functional>
#include <list>
#include <unordered_map>
#include "Command.hpp"

namespace adas
{
    // using Cmder = std::function<void(PoseHandler &poseHandler)>;
    // using Cmder = std::function<ActionGroup(PoseHandler &poseHandler)>;
    using Cmder = std::function<ActionGroup(PoseHandler &poseHandler, const CmderOrchestrator &orchestrator)>;
    using CmderList = std::list<Cmder>;

    class CmderFactory final
    {
    public:
        CmderFactory(void) = default;
        ~CmderFactory(void) = default;

        CmderFactory(const CmderFactory &) = delete;
        CmderFactory &operator=(const CmderFactory &) = delete;

    public:
        // std::list<std::function<void(PoseHandler& poseHandler)>> GetCmders(const std::string& command) const noexcept;
        CmderList GetCmders(const std::string &command) const noexcept;

    private:
        // const std::unordered_map<char, std::function<void(PoseHandler & PoseHandler)>> cmderMap {
        //     {'M', MoveCommand()},
        //     {'L', TurnLeftCommand()},
        //     {'R', TurnRightCommand()},
        //     {'F', FastCommand()},
        //     {'B', ReverseCommand()}
        // };

        std::string ParseCommandString(std::string_view command) const noexcept;
        void ReplaceAll(std::string &inout,std::string_view what, std::string_view with) const noexcept;

        const std::unordered_map<char, Cmder> cmderMap {
            {'M', MoveCommand()},
            {'L', TurnLeftCommand()},
            {'R', TurnRightCommand()},
            {'F', FastCommand()},
            {'B', ReverseCommand()},
            {'Z', TurnRoundCommand()}
        };
    };
}