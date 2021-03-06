#include "checkpoint/checkpoint.h"
#include "moleculardynamics/Ar_moleculardynamics.h"

namespace {
    static auto const LOOP = 100;
}

int main()
{
    checkpoint::CheckPoint cp;

    cp.checkpoint("処理開始", __LINE__);

    moleculardynamics::Ar_moleculardynamics armd;
    
    cp.checkpoint("初期化処理", __LINE__);

    for (auto i = 0; i < LOOP; i++) {
        armd.Calc_Forces();
        armd.Move_Atoms();
    }

    cp.checkpoint("CUDAで並列化", __LINE__);

    cp.checkpoint_print();

    return 0;
}
