// Does Z3's built-in parallel mode speed up a single hard check()?
#include <z3.h>
#include <vector>
#include <random>
#include <cstdio>
#include <chrono>
using namespace std::chrono;
static double ms(steady_clock::time_point a,steady_clock::time_point b){
  return duration<double,std::milli>(b-a).count(); }

int main(int argc, char** argv){
  int V     = argc>1?atoi(argv[1]):140;
  int CL    = argc>2?atoi(argv[2]):600;   // clauses (ratio ~4.3 -> hard)
  std::mt19937 rng(999);
  std::uniform_int_distribution<int> var(0,V-1), sgn(0,1);
  struct C{int l[3];}; std::vector<C> cls;
  for(int i=0;i<CL;++i){ C c; int u[3]={-1,-1,-1};
    for(int k=0;k<3;++k){int v;do{v=var(rng);}while(v==u[0]||v==u[1]);u[k]=v;
      c.l[k]=(sgn(rng)?1:-1)*(v+1);} cls.push_back(c);}

  auto run=[&](bool par, int threads)->double{
    char t[8]; snprintf(t,sizeof(t),"%d", par?threads:1);
    Z3_global_param_set("sat.threads", t);   // Z3's internal parallel SAT
    Z3_config cfg=Z3_mk_config();
    Z3_context c=Z3_mk_context(cfg); Z3_del_config(cfg);
    Z3_solver s=Z3_mk_solver(c); Z3_solver_inc_ref(c,s);
    for(auto&cl:cls){ Z3_ast a[3];
      for(int k=0;k<3;++k){ int l=cl.l[k]; char nm[16]; snprintf(nm,sizeof(nm),"x%d",l<0?-l:l);
        Z3_ast v=Z3_mk_const(c,Z3_mk_string_symbol(c,nm),Z3_mk_bool_sort(c));
        a[k]= l<0?Z3_mk_not(c,v):v; }
      Z3_solver_assert(c,s,Z3_mk_or(c,3,a)); }
    auto t0=steady_clock::now(); int r=Z3_solver_check(c,s); double d=ms(t0,steady_clock::now());
    printf("  %-22s %8.1f ms   %s\n", par?("sat.threads="+std::to_string(threads)).c_str():"sat.threads=1 (seq)",
           d, r==Z3_L_TRUE?"SAT":r==Z3_L_FALSE?"UNSAT":"?");
    Z3_solver_dec_ref(c,s); Z3_del_context(c); return d;
  };
  printf("V=%d CL=%d (single hard check)\n", V, CL);
  run(false,1);
  run(true,4);
  run(true,8);
  return 0;
}
