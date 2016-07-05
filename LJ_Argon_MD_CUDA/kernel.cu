#include "checkpoint/checkpoint.h"
#include "moleculardynamics/Ar_moleculardynamics.h"

namespace {
    static auto const LOOP = 100;
}

int main()
{
    checkpoint::CheckPoint cp;

    cp.checkpoint("�����J�n", __LINE__);

    moleculardynamics::Ar_moleculardynamics armd;
    
    cp.checkpoint("����������", __LINE__);

    for (auto i = 0; i < LOOP; i++) {
        armd.Calc_Forces();
        armd.Move_Atoms();
    }

    cp.checkpoint("CUDA�ŕ���", __LINE__);

    cp.checkpoint_print();

    return 0;
}
