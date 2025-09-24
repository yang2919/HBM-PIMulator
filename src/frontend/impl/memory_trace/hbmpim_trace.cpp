#include <filesystem>
#include <iostream>
#include <fstream>

#include "frontend/frontend.h"
#include "base/exception.h"

namespace Ramulator
{

  namespace fs = std::filesystem;

  class HBMPIMTrace : public IFrontEnd, public Implementation
  {
    RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, HBMPIMTrace, "HBMPIMTrace", "HBM-PIM access address trace.")

  private:
    struct Trace
    {
      int mode;
      int opcode;
      Addr_t addr;
      POperand_t poperand;
    };
    std::vector<Trace> m_trace;

    size_t m_trace_length = 0;
    size_t m_curr_trace_idx = 0;
    bool m_trace_end = false;

    size_t m_trace_count = 0;

    Logger_t m_logger;

    std::map<std::string, int> str_to_ISR;
    std::map<std::string, int> str_to_mode;
    std::map<std::string, LOCATE> str_to_Loc;

    bool trace_reached_callback = false;
    std::function<void(Request &)> callback;

  public:
    void init() override
    {
      std::string trace_path_str = param<std::string>("path").desc("Path to the load store trace file.").required();
      m_clock_ratio = param<uint>("clock_ratio").required();

      str_to_DEF();

      m_logger = Logging::create_logger("HBMPIMTrace");
      m_logger->info("Loading trace file {} ...", trace_path_str);
      init_trace(trace_path_str);
      m_logger->info("Loaded {} lines.", m_trace.size());
      callback = std::bind(&HBMPIMTrace::receive, this, std::placeholders::_1);
    };

    void tick() override
    {
      if(!m_trace_end){
        const Trace &t = m_trace[m_curr_trace_idx];
        Request req({t.addr, t.mode, t.opcode, t.poperand});
        if(req.operation_id == Opcode::EXIT) req.callback = callback;
        bool request_sent = m_memory_system->send(req);

        if (request_sent)
        {
          m_curr_trace_idx += 1;
          if (m_curr_trace_idx == m_trace_length) m_trace_end = true;
          m_trace_count++;
        }
      }
    };

  private:
    void init_trace(const std::string &file_path_str)
    {
      fs::path trace_path(file_path_str);
      if (!fs::exists(trace_path))
      {
        throw ConfigurationError("Trace {} does not exist!", file_path_str);
      }

      std::ifstream trace_file(trace_path);
      if (!trace_file.is_open())
      {
        throw ConfigurationError("Trace {} cannot be opened!", file_path_str);
      }

      std::string line;
      while (std::getline(trace_file, line))
      {
        if (line[0] == '#' || line.empty())
        {
          continue; // comment or empty line
        }
        std::vector<std::string> tokens;
        tokenize(tokens, line, " ");

        int mode;
        int opcode;
        Addr_t addr;
        POperand_t poperand;

        mode = str_to_mode[tokens[0]];
        if (mode == Request::Type::Read || mode == Request::Type::Write)
        {
          addr = std::stoll(tokens[1]);
        }
        else
        {
          opcode = str_to_ISR[tokens[1]];

          if (mode == Request::Type::PIM)
          {
            for (int i = 2; i < tokens.size(); i++)
            {
              std::vector<std::string> token;
              tokenize(token, tokens[i], ",");

              LOCATE loc = str_to_Loc[token[0]];
              int ad1 = std::stoi(token[1]);
              int ad2 = -1;
              int ad3 = -1;

              if (loc == LOCATE::BANK)
              {
                ad2 = std::stoi(token[2]);
                ad3 = std::stoi(token[3]);
              }
              poperand.push_back({loc, ad1, ad2, ad3});
            }
          }
        }

        m_trace.push_back({mode, opcode, addr, poperand});
      }

      trace_file.close();
      m_trace_length = m_trace.size();
    };

    void receive(Request &req)
    {
      trace_reached_callback = true;
    }

    // TODO: FIXME
    bool is_finished() override
    {
      return trace_reached_callback;
    };

    void str_to_DEF()
    {
      str_to_ISR["ADD"] = Opcode::ADD;
      str_to_ISR["MUL"] = Opcode::MUL;
      str_to_ISR["MAC"] = Opcode::MAC;
      str_to_ISR["MAD"] = Opcode::MAD;
      str_to_ISR["MOV"] = Opcode::MOV;
      str_to_ISR["FILL"] = Opcode::FILL;
      str_to_ISR["NOP"] = Opcode::NOP;
      str_to_ISR["JUMP"] = Opcode::JUMP;
      str_to_ISR["EXIT"] = Opcode::EXIT;

      str_to_mode["R"] = Request::Type::Read;
      str_to_mode["W"] = Request::Type::Write;
      str_to_mode["AB"] = Request::Type::AB;
      str_to_mode["PIM"] = Request::Type::PIM;

      str_to_Loc["BANK"] = LOCATE::BANK;
      str_to_Loc["GRF"] = LOCATE::GRF;
      str_to_Loc["SRF"] = LOCATE::SRF;
    }
  };

} // namespace Ramulator