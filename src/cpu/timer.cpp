#include <gpubf/timer.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace std;

namespace ccd {

struct Record
{
      Timer timer;
      char * tag;
      json j_object;

      Record(){};

      void Start(char * s)
      {
            tag = s;
            timer.start();
      }

      void Start(char * s, json & jtmp)
      {
           j_object = jtmp;
           Start(s);
      } 

      void Stop()
      {
            timer.stop();
            double elapsed = 0;
            elapsed += timer.getElapsedTimeInMicroSec();
            // j_object[tag]=elapsed;
            j_object.push_back(json::object_t::value_type(tag, elapsed));
            printf("%s : %.3f ms\n", tag, elapsed / 1000.f);
      }

      void Print()
      {
            cout << j_object.dump() << endl;  
      }
      
      json Dump()
      {
            return j_object;
      }
};


}