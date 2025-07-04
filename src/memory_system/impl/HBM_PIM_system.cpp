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

    void apply_addr_mapp(Request &req, int channel_id) {
        req.addr_vec.resize(5, -1);

        req.addr_vec[0] = channel_id;
        req.addr_vec[1] = 0;
        req.addr_vec[2] = 0;
        
        req.addr_vec[3] = 0;
        req.addr_vec[4] = 0;
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

    bool send_all(Request req){
      for (int i = 0; i < 16; i++) {
        Request pim_req = req;
        for (int cnt = 0; cnt < m_controllers.size(); cnt++) {
          apply_addr_mapp(pim_req, cnt);
          // m_logger->info("[CLK {}] 1- Sending {} to channel {}", m_clk, aim_req.str(), channel_id);
          if (m_controllers[cnt]->send(pim_req) == false) {
            return false;
          }
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

