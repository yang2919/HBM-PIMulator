#include "memory_system/memory_system.h"
#include "translation/translation.h"
#include "dram_controller/controller.h"
#include "addr_mapper/addr_mapper.h"
#include "dram/dram.h"

namespace Ramulator {

class HBMPIMSystem  final : public IMemorySystem, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, HBMPIMSystem, "HBMPIMSystem", "A HBM-PIM-based memory system.");

  protected:
    Clk_t m_clk = 0;
    IDRAM*  m_dram;
    IAddrMapper*  m_addr_mapper;
    std::vector<IDRAMController*> m_controllers;

    struct Mode{
      enum : int{
        SB,
        AB,
        PIM
      };
    };

    int current_mode = Mode::SB;

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

    bool send_all(Request req, int& request_cnt){
      for (int channel_id = 0; channel_id < m_controllers.size(); channel_id++) {
        // m_logger->info("[CLK {}] 2- Sending {} to channel {}", m_clk, aim_req.str(), channel_id);
        if (m_controllers[channel_id]->send(req) == false) {
          return false;
        }
      }
      return true;
    }

    bool send(Request req) override {
      // SB operation
      if (req.type_id == Request::Type::Read || req.type_id == Request::Type::Write){ // Type Transition
        if (current_mode != Mode::SB){
          if (current_mode == Mode::PIM){
            Request r = Request(Opcode::TMOD_P);
            if(send_all(r, s_num_trans_requests) == false) return false;
            current_mode = Mode::AB;
          }
          if (current_mode == Mode::AB){
            Request r = Request(Opcode::TMOD_A);
            if(send_all(r, s_num_trans_requests) == false) return false;
            current_mode = Mode::SB;
          }
        }
        
        m_addr_mapper->apply(req);
        int channel_id = req.addr_vec[0];
        bool is_success = m_controllers[channel_id]->send(req);

        if (is_success) {
          switch (req.operation_id) {
            case Opcode::READ: {
              s_num_read_requests++;
              break;
            }
            case Opcode::WRITE: {
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
            if(send_all(r, s_num_trans_requests) == false) return false;
            current_mode = Mode::AB;
          } else if(current_mode == Mode::PIM){
            Request r = Request(Opcode::TMOD_P);
            if(send_all(r, s_num_trans_requests) == false) return false;
            current_mode = Mode::AB;
          }
        }

        if(send_all(req, s_num_write_requests) == false) return false;
      } 
      // PIM Operation
      else if (req.type_id == Request::Type::PIM){
        if(current_mode != Mode::PIM){
          if(current_mode == Mode::SB){
            Request r = Request(Opcode::TMOD_A);
            if(send_all(r, s_num_trans_requests) == false) return false;
            current_mode = Mode::AB;
          }
          if(current_mode == Mode::AB){
            Request r = Request(Opcode::TMOD_A);
            if(send_all(r, s_num_trans_requests) == false) return false;
            current_mode = Mode::PIM;
          }
        }

        if(send_all(req, s_num_pim_requests) == false) return false;
      }

      return true;
    };
    
    void tick() override {
      m_clk++;
      m_dram->tick();
      for (auto controller : m_controllers) {
        controller->tick();
      }
    };

    float get_tCK() override {
      return m_dram->m_timing_vals("tCK_ps") / 1000.0f;
    }

    // const SpecDef& get_supported_requests() override {
    //   return m_dram->m_requests;
    // };
};
  
}   // namespace 

