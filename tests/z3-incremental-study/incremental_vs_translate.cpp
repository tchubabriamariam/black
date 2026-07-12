// Raw-Z3 micro-benchmark: value of incremental reuse, and cost of translate.
//
// Growing hard SAT instance (random 3-SAT near phase transition), asserted in
// batches ("depths"). Three strategies:
//   INC    one solver, assert batch + check each depth        (reuses learning)
//   COLD   fresh solver rebuilt from scratch at each depth     (parallel thread)
//   XLATE  build spine once; translate a copy per depth to check (proposed share)
#include <z3.h>
#include <vector>
#include <random>
#include <cstdio>
#include <chrono>

using namespace std::chrono;
static double ms(steady_clock::time_point a, steady_clock::time_point b){
  return duration<double,std::milli>(b-a).count();
}

struct Clause { int l[3]; }; // literals: (var+1) or -(var+1)

int main(int argc, char** argv){
  int V     = argc>1?atoi(argv[1]):55;   // variables
  int DEPTH = argc>2?atoi(argv[2]):16;   // number of depths (batches)
  int BATCH = argc>3?atoi(argv[3]):14;   // clauses added per depth
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> var(0,V-1), sign(0,1);

  // Pre-generate the whole clause sequence (shared by all strategies).
  std::vector<Clause> cls;
  for(int d=0; d<DEPTH; ++d)
    for(int b=0; b<BATCH; ++b){
      Clause c; int used[3]={-1,-1,-1};
      for(int i=0;i<3;++i){ int v; do{v=var(rng);}while(v==used[0]||v==used[1]); used[i]=v;
        c.l[i]=(sign(rng)?1:-1)*(v+1); }
      cls.push_back(c);
    }

  auto mk_ctx = []{ Z3_config cfg=Z3_mk_config(); Z3_context c=Z3_mk_context(cfg); Z3_del_config(cfg); return c; };
  auto lit = [](Z3_context c, int l)->Z3_ast{
    char nm[16]; snprintf(nm,sizeof(nm),"x%d", l<0?-l:l);
    Z3_ast v = Z3_mk_const(c, Z3_mk_string_symbol(c,nm), Z3_mk_bool_sort(c));
    return l<0 ? Z3_mk_not(c,v) : v;
  };
  auto assert_clause = [&](Z3_context c, Z3_solver s, const Clause& cl){
    Z3_ast a[3]={lit(c,cl.l[0]),lit(c,cl.l[1]),lit(c,cl.l[2])};
    Z3_solver_assert(c,s,Z3_mk_or(c,3,a));
  };

  int last_res = -2;

  // ---- INC: one solver, incremental ----
  auto t0=steady_clock::now();
  { Z3_context c=mk_ctx(); Z3_solver s=Z3_mk_solver(c); Z3_solver_inc_ref(c,s);
    for(int d=0; d<DEPTH; ++d){
      for(int b=0;b<BATCH;++b) assert_clause(c,s,cls[d*BATCH+b]);
      last_res=Z3_solver_check(c,s);
    }
    Z3_solver_dec_ref(c,s); Z3_del_context(c);
  }
  double inc_ms=ms(t0,steady_clock::now());

  // ---- COLD: rebuild fresh each depth ----
  t0=steady_clock::now();
  for(int d=0; d<DEPTH; ++d){
    Z3_context c=mk_ctx(); Z3_solver s=Z3_mk_solver(c); Z3_solver_inc_ref(c,s);
    for(int j=0;j<=d;++j) for(int b=0;b<BATCH;++b) assert_clause(c,s,cls[j*BATCH+b]);
    Z3_solver_check(c,s);
    Z3_solver_dec_ref(c,s); Z3_del_context(c);
  }
  double cold_ms=ms(t0,steady_clock::now());

  // ---- XLATE: spine built once; translate a copy per depth to check ----
  double xlate_only=0;
  t0=steady_clock::now();
  { Z3_context spine=mk_ctx(); Z3_solver ss=Z3_mk_solver(spine); Z3_solver_inc_ref(spine,ss);
    for(int d=0; d<DEPTH; ++d){
      for(int b=0;b<BATCH;++b) assert_clause(spine,ss,cls[d*BATCH+b]);
      auto x0=steady_clock::now();
      Z3_context wc=mk_ctx();
      Z3_solver ws=Z3_solver_translate(spine,ss,wc); Z3_solver_inc_ref(wc,ws);
      xlate_only+=ms(x0,steady_clock::now());
      Z3_solver_check(wc,ws);                 // the per-depth check, on the copy
      Z3_solver_dec_ref(wc,ws); Z3_del_context(wc);
    }
    Z3_solver_dec_ref(spine,ss); Z3_del_context(spine);
  }
  double xlate_ms=ms(t0,steady_clock::now());

  printf("V=%d depths=%d batch=%d  final=%s\n", V,DEPTH,BATCH,
         last_res==Z3_L_TRUE?"SAT":last_res==Z3_L_FALSE?"UNSAT":"?");
  printf("  INC   (incremental, 1 solver)      %8.1f ms\n", inc_ms);
  printf("  COLD  (rebuild each depth)         %8.1f ms   (%.2fx INC)\n", cold_ms, cold_ms/inc_ms);
  printf("  XLATE (translate copy each depth)  %8.1f ms   (%.2fx INC)  translate-only=%.1f ms\n",
         xlate_ms, xlate_ms/inc_ms, xlate_only);
  return 0;
}
