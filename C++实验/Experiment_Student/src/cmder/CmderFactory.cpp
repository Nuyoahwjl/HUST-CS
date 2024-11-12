#include "CmderFactory.hpp"

namespace adas
{
    // std::list<std::function<void(PoseHandler & PoseHandler)>> CmderFactory::GetCmders(const std::string &command) const noexcept
    // {
    //     std::list<std::function<void(PoseHandler & PoseHandler)>> cmders;
    //     for (const auto cmd : command)
    //     {
    //         auto iter = cmderMap.find(cmd);
    //         if (iter != cmderMap.end())
    //         {
    //             cmders.push_back(iter->second);
    //         }
    //     }
    //     return cmders;
    // }
    CmderList CmderFactory::GetCmders(const std::string &command) const noexcept
    {
        CmderList cmders;
        // for (const auto cmd : command)
        for (const auto cmd : ParseCommandString(command))
        {
            auto iter = cmderMap.find(cmd);
            if (iter != cmderMap.end())
            {
                cmders.push_back(iter->second);
            }
        }
        return cmders;
    }

    std::string CmderFactory::ParseCommandString(std::string_view command) const noexcept
    {
        std::string result(command);
        ReplaceAll(result, "TR", "Z");
        return result;
    }

    void CmderFactory::ReplaceAll(std::string &inout, std::string_view what, std::string_view with) const noexcept
    {
        for(
            std::string::size_type pos{};
            inout.npos != (pos = inout.find(what.data(), pos, what.length()));
            pos += with.length()
        ) {
            inout.replace(pos, what.length(), with.data(), with.length());
        }
    }

}