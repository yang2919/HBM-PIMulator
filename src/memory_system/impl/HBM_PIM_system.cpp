#include "memory_system/memory_system.h"
#include "translation/translation.h"
#include "dram_controller/controller.h"
#include "addr_mapper/addr_mapper.h"
#include "dram/dram.h"

namespace Ramulator {

class HBMPIMSystem  final : public IMemorySystem, public Implementation {

  #define REQ_SIZE 1<<21
  #define MAX_CHANNEL_COUNT 32

  RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, HBMPIMSystem, "HBMPIMSystem", "A HBM-PIM-based memory system.");

  protected:
    Clk_t m_clk = 0;
    IDRAM*  m_dram;
    IAddrMapper*  m_addr_mapper;
    std::vector<IDRAMController*> m_controllers;

    std::queue<Request> request_queue;
    std::queue<Request> remaining_requests[MAX_CHANNEL_COUNT];

    bool finished = false;

    struct Mode{
      enum : int{
        SB,
        AB,
        PIM
      };
    };

    int current_mode = Mode::SB;

    void apply_addr_mapp(Request &req, int channel_id) {
        req.addr_vec.resize(5, -1);

        req.addr_vec[0] = channel_id;
        req.addr_vec[1] = 0;
        req.addr_vec[2] = 0;
        
        req.addr_vec[3] = 0;
        if(req.poperand.size() == 0)
          req.addr_vec[4] = 0;
        else {
          if(req.poperand[0].loc == LOCATE::BANK){
              req.addr_vec[4] = req.poperand[0].addr;
          }
          else if(req.poperand[1].loc == LOCATE::BANK){
              req.addr_vec[4] = req.poperand[1].addr;
          }
        }
    }

  public:
    int s_num_read_requests = 0;
    int s_num_write_requests = 0;
    int s_num_pim_requests = 0;
    int s_num_trans_requests = 0;

  public:
    void init() override { 
      // Create device (a top-level node wrapping all channel nodes)
      m_dram = create_child_ifce<IDRAM>();
      m_addr_mapper = create_child_ifce<IAddrMapper>();

      int num_channels = m_dram->get_level_size("channel");   

      // Create memory controllers
      for (int i = 0; i < num_channels; i++) {
        IDRAMController* controller = create_child_ifce<IDRAMController>();
        controller->m_impl->set_id(fmt::format("Channel {}", i));
        controller->m_channel_id = i;
        m_controllers.push_back(controller);
      }

      m_clock_ratio = param<uint>("clock_ratio").required();

      register_stat(m_clk).name("memory_system_cycles");
      register_stat(s_num_read_requests).name("total_num_read_requests");
      register_stat(s_num_write_requests).name("total_num_write_requests");
      register_stat(s_num_pim_requests).name("total_num_pim_requests");
      register_stat(s_num_trans_requests).name("total_num_trans_requests");
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override { }

    bool send(Request req) override {
          request_queue.push(req);
          // m_logger->info("[CLK {}] {} pushed to the queue!", m_clk, req.str());

          switch (req.type_id) {
            case Request::Type::PIM: {
                s_num_pim_requests++;
                break;
            }
            case Request::Type::Read: {
                s_num_read_requests++;
                break;
            }
            case Request::Type::Write: {
                s_num_write_requests++;
                break;
            }
            default: {
                throw ConfigurationError("DRAMSystem: unknown request type {}!", (int)req.type_id);
                break;
            }
        }

        return true;
    };
/*
    bool send(Request req) override { // Todo: seperate timing of transition and PIM commands.
      // SB operation
      if (req.type_id == Request::Type::Read || req.type_id == Request::Type::Write){ // Type Transition
        if (current_mode != Mode::SB){
          if (current_mode == Mode::PIM){
            Request r = Request(Opcode::TMOD_P);
            if(send_all(r) == false) return false;
            s_num_trans_requests++;
            current_mode = Mode::AB;
          }
          if (current_mode == Mode::AB){
            Request r = Request(Opcode::TMOD_A);
            if(send_all(r) == false) return false;
            s_num_trans_requests++;
            current_mode = Mode::SB;
          }
        }
        
        m_addr_mapper->apply(req);
        int channel_id = req.addr_vec[0];
        bool is_success = m_controllers[channel_id]->send(req);

        if (is_success) {
          switch (req.type_id) {
            case Request::Type::Read: {
              s_num_read_requests++;
              break;
            }
            case Request::Type::Write: {
              s_num_write_requests++;
              break;
            }
          }
        }
      } 
      // AB Operation
      else if (req.type_id == Request::Type::AB){
        if(current_mode != Mode::AB){
          if(current_mode == Mode::SB){
            Request r = Request(Opcode::TMOD_A);
            if(send_all(r) == false) return false;
            s_num_trans_requests++;
            current_mode = Mode::AB;
          } else if(current_mode == Mode::PIM){
            Request r = Request(Opcode::TMOD_P);
            if(send_all(r) == false) return false;
            s_num_trans_requests++;
            current_mode = Mode::AB;
          }
        }

        if(send_all(req) == false) return false;
        s_num_write_requests++;
      } 
      // PIM Operation
      else if (req.type_id == Request::Type::PIM){
        if(current_mode != Mode::PIM){
          if(current_mode == Mode::SB){
            Request r = Request(Opcode::TMOD_A);
            if(send_all(r) == false) return false;
            s_num_trans_requests++;
            current_mode = Mode::AB;
          }
          
          if(current_mode == Mode::AB){
            Request r = Request(Opcode::TMOD_A);
            if(send_all(r) == false) return false;
            s_num_trans_requests++;
            current_mode = Mode::PIM;
          }
        }

        if(send_all(req) == false) return false;
        s_num_pim_requests++;
      }
      
      return true;
    }; */
    
    void tick() override {
      bool was_request_remaining = false;
      for (int channel_id = 0; channel_id < MAX_CHANNEL_COUNT; channel_id++) {
          while (remaining_requests[channel_id].empty() == false) {
              was_request_remaining = true;
              // m_logger->info("[CLK {}] 0- Sending {} to channel {}", m_clk, remaining_AiM_requests[channel_id].front().str(), channel_id);
              if (m_controllers[channel_id]->send(remaining_requests[channel_id].front()) == false) {
                  // m_logger->info("[CLK {}] 0- failed", m_clk, channel_id);
                  break;
              }
              remaining_requests[channel_id].pop();
          }
      }

      if(was_request_remaining == false){
        if(request_queue.empty() == true){
          finished = true;
        }
        else{
          Request req = request_queue.front();
          request_queue.pop();
          if (req.type_id == Request::Type::Read || req.type_id == Request::Type::Write){ // Type Transition
            if (current_mode != Mode::SB){
              if (current_mode == Mode::PIM){
                Request r = Request(Opcode::TMOD_P);
                for (int cnt = 0; cnt < m_controllers.size(); cnt++) {
                  apply_addr_mapp(r, cnt);
                  // m_logger->info("[CLK {}] 1- Sending {} to channel {}", m_clk, aim_req.str(), channel_id);
                  if (m_controllers[cnt]->send(r) == false) {
                    remaining_requests[cnt].push(r);
                  }
                }
                
                s_num_trans_requests++;
                current_mode = Mode::AB;
              }
              if (current_mode == Mode::AB){
                Request r = Request(Opcode::TMOD_A);
                for (int cnt = 0; cnt < m_controllers.size(); cnt++) {
                  apply_addr_mapp(r, cnt);
                  // m_logger->info("[CLK {}] 1- Sending {} to channel {}", m_clk, aim_req.str(), channel_id);
                  if (m_controllers[cnt]->send(r) == false) {
                    remaining_requests[cnt].push(r);
                  }
                }

                s_num_trans_requests++;
                current_mode = Mode::SB;
              }
            }
            
            m_addr_mapper->apply(req);
            int channel_id = req.addr_vec[0];
            if(m_controllers[channel_id]->send(req) == false){
              remaining_requests[channel_id].push(req);
            }
          } 

          // AB Operation
          else if (req.type_id == Request::Type::AB){
            if(current_mode != Mode::AB){
              if(current_mode == Mode::SB){
                Request r = Request(Opcode::TMOD_A);
                for (int cnt = 0; cnt < m_controllers.size(); cnt++) {
                  apply_addr_mapp(r, cnt);
                  // m_logger->info("[CLK {}] 1- Sending {} to channel {}", m_clk, aim_req.str(), channel_id);
                  if (m_controllers[cnt]->send(r) == false) {
                    remaining_requests[cnt].push(r);
                  }
                }

                s_num_trans_requests++;
                current_mode = Mode::AB;
              } else if(current_mode == Mode::PIM){
                Request r = Request(Opcode::TMOD_P);
                for (int cnt = 0; cnt < m_controllers.size(); cnt++) {
                  apply_addr_mapp(r, cnt);
                  // m_logger->info("[CLK {}] 1- Sending {} to channel {}", m_clk, aim_req.str(), channel_id);
                  if (m_controllers[cnt]->send(r) == false) {
                    remaining_requests[cnt].push(r);
                  }
                }

                s_num_trans_requests++;
                current_mode = Mode::AB;
              }
            }

            for (int cnt = 0; cnt < m_controllers.size(); cnt++) {
              apply_addr_mapp(req, cnt);
              // m_logger->info("[CLK {}] 1- Sending {} to channel {}", m_clk, aim_req.str(), channel_id);
              if (m_controllers[cnt]->send(req) == false) {
                remaining_requests[cnt].push(req);
              }
            }
            s_num_write_requests++;
          } 
          // PIM Operation
          else if (req.type_id == Request::Type::PIM){
            if(current_mode != Mode::PIM){
              if(current_mode == Mode::SB){
                Request r = Request(Opcode::TMOD_A);
                for (int cnt = 0; cnt < m_controllers.size(); cnt++) {
                  apply_addr_mapp(r, cnt);
                  // m_logger->info("[CLK {}] 1- Sending {} to channel {}", m_clk, aim_req.str(), channel_id);
                  if (m_controllers[cnt]->send(r) == false) {
                    remaining_requests[cnt].push(r);
                  }
                }

                s_num_trans_requests++;
                current_mode = Mode::AB;
              }
              
              if(current_mode == Mode::AB){
                Request r = Request(Opcode::TMOD_A);
                for (int cnt = 0; cnt < m_controllers.size(); cnt++) {
                  apply_addr_mapp(r, cnt);
                  // m_logger->info("[CLK {}] 1- Sending {} to channel {}", m_clk, aim_req.str(), channel_id);
                  if (m_controllers[cnt]->send(r) == false) {
                    remaining_requests[cnt].push(r);
                  }
                }

                s_num_trans_requests++;
                current_mode = Mode::PIM;
              }
            }

            for (int cnt = 0; cnt < m_controllers.size(); cnt++) {
              apply_addr_mapp(req, cnt);
              // m_logger->info("[CLK {}] 1- Sending {} to channel {}", m_clk, aim_req.str(), channel_id);
              if (m_controllers[cnt]->send(req) == false) {
                remaining_requests[cnt].push(req);
              }
            }
          }
        }
      }

      m_clk++;
      m_dram->tick();
      
      for (auto controller : m_controllers) {
        controller->tick();
      }
    };

    float get_tCK() override {
      return m_dram->m_timing_vals("tCK_ps") / 1000.0f;
    }

    bool is_finished()
    {
      return finished;
    }

    // const SpecDef& get_supported_requests() override {
    //   return m_dram->m_requests;
    // };
};
  
}   // namespace 

