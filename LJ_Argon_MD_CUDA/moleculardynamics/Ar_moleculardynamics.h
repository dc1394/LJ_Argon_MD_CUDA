/*! \file Ar_moleculardynamics.h
    \brief アルゴンに対して、分子動力学シミュレーションを行うクラスの宣言

    Copyright ©  2015 @dc1394 All Rights Reserved.
    This software is released under the BSD 2-Clause License.
*/

#ifndef _AR_MOLECULARDYNAMICS_H_
#define _AR_MOLECULARDYNAMICS_H_

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../myrandom/myrand.h"
#include <cstdint>                                  // for std::int32_t
#include <cmath>                                    // for std::sqrt, std::pow
#include <iostream>                                 // for std::ios_base::fixed, std::ios_base::floatfield, std::cout
#include <memory>
#include <boost/format.hpp>
#include <boost/range/algorithm/generate.hpp>       // for boost::generate
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

namespace moleculardynamics {
    //! A function.
    /*!
        原子に働く力を計算する（コア部分）
        \param rv 各原子の座標
        \param F 各原子に働く力（出力）
        \param Up 各原子に働くポテンシャルエネルギー
        \param ncp 相互作用を計算するセルの個数
    */
    __global__ void Calc_Forces_Core(
        float4 const * rv,
        float4 * F,
        float * Up,
        std::int32_t const ncp,
        std::int32_t const numatom,
        float const periodiclen,
        float const rc2,
        float const Vrc);

    //! A function.
    /*!
        周期境界条件をチェックする
        \param r 各原子の座標
        \param r1 原子の初期座標
        \param periodiclen 周期境界条件の長さ
    */
    __global__ void check_periodic(
        float4 * r,
        float4 * r1,
        float const periodiclen);

    //! A function.
    /*!
        原子を移動させる（第一ステップ、コア部分）
    */
    __global__ void Move_Atoms1_Core(
        float4 const * F,
        float4 * r,
        float4 * r1,
        float4 * V,
        float const dt2,
        float const s);

    //! A function.
    /*!
        原子を移動させる（コア部分）
    */
    __global__ void Move_Atoms_Core(
        float4 const * F,
        float4 * r,
        float4 * r1,
        float4 * V,
        float const dt,
        float const s);

    //! A functional.
    /*!
        ベクトルの大きさの2乗を計算する関数オブジェクト
    */
    struct norm2 {
        __host__ __device__ float operator()(float4 const & v) const
        {
            return v.x * v.x + v.y * v.y + v.z * v.z;
        }
    };

    //! A class.
    /*!
        アルゴンに対して、分子動力学シミュレーションを行うクラス
    */
    template <typename T>
    class Ar_moleculardynamics final {
        // #region コンストラクタ・デストラクタ

    public:
        //! A constructor.
        /*!
            コンストラクタ
        */
        Ar_moleculardynamics();

        //! A destructor.
        /*!
            デフォルトデストラクタ
        */
        ~Ar_moleculardynamics() = default;

        // #endregion コンストラクタ・デストラクタ

        // #region publicメンバ関数

        //! A public member function.
        /*!
            原子に働く力を計算する
        */
        void Calc_Forces();

        //! A public member function.
        /*!
            原子を移動させる
        */
        void Move_Atoms();
        
        //! A public member function.
        /*!
            初期化する
        */
        void reset();

        // #endregion publicメンバ関数

        // #region privateメンバ関数

    private:
        //! A private member function.
        /*!
            原子の初期位置を決める
        */
        void MD_initPos();

        //! A private member function.
        /*!
            原子の初期速度を決める
        */
        void MD_initVel();

        // #endregion privateメンバ関数

        // #region メンバ変数

    public:
        //! A private member variable (constant).
        /*!
            初期のスーパーセルの個数
        */
        static auto const FIRSTNC = 3;

        //! A private member variable (constant).
        /*!
            初期の格子定数のスケール
        */
        static T const FIRSTSCALE;

        //! A private member variable (constant).
        /*!
            初期温度（絶対温度）
        */
        static T const FIRSTTEMP;

    private:
        //! A private member variable (constant).
        /*!
            Woodcockの温度スケーリングの係数
        */
        static T const ALPHA;

        //! A private member variable (constant).
        /*!
            時間刻みΔt
        */
        static T const DT;
        
        //! A private member variable (constant).
        /*!
            ボルツマン定数
        */
        static T const KB;
        
        //! A private member variable (constant).
        /*!
            SMの個数
        */
        static auto const SMNUM = 12;

        //! A private member variable (constant).
        /*!
            アルゴン原子に対するε
        */
        static T const YPSILON;

        //! A private member variable (constant).
        /*!
            時間刻みの二乗
        */
        T const dt2;
                
        //! A private member variable.
        /*!
            格子定数
        */
        T lat_;

        //! A private member variable (constant).
        /*!
            スーパーセルの個数
        */
        std::int32_t Nc_ = Ar_moleculardynamics::FIRSTNC;
        
        //! A private member variable.
        /*!
            n個目の原子に働く力（デバイス側）
        */
        thrust::device_vector<float4> F_dev_;
        
        //! A private member variable.
        /*!
            MDのステップ数
        */
        std::int32_t MD_iter_;

        //! A private member variable (constant).
        /*!
            相互作用を計算するセルの個数
        */
        std::int32_t const ncp_ = 3;

        //! A private member variable.
        /*!
            原子数
        */
        std::int32_t NumAtom_;
        
        //! A private member variable.
        /*!
            周期境界条件の長さ
        */
        T periodiclen_;
                
        //! A private member variable.
        /*!
            n個目の原子の座標（デバイス側）
        */
        thrust::device_vector<float4> r_dev_;

        //! A private member variable.
        /*!
            n個目の原子の初期座標（デバイス側）
        */
        thrust::device_vector<float4> r1_dev_;
        
        //! A private member variable (constant).
        /*!
            カットオフ半径
        */
        T const rc_ = 2.5;

        //! A private member variable (constant).
        /*!
            カットオフ半径の2乗
        */
        T const rc2_;

        //! A private member variable (constant).
        /*!
            カットオフ半径の逆数の6乗
        */
        T const rcm6_;

        //! A private member variable (constant).
        /*!
            カットオフ半径の逆数の12乗
        */
        T const rcm12_;

        //! A private member variable.
        /*!
            格子定数のスケーリングの定数
        */
        T scale_ = Ar_moleculardynamics::FIRSTSCALE;

        //! A private member variable.
        /*!
            計算された温度Tcalc
        */
        T Tc_;

        //! A private member variable.
        /*!
            与える温度Tgiven
        */
        T Tg_;
        
        //! A private member variable.
        /*!
            運動エネルギー
        */
        T Uk_;

        //! A private member variable.
        /*!
            ポテンシャルエネルギー
        */
        T Up_;

        //! A private member variable.
        /*!
            各原子のポテンシャルエネルギー（デバイス側）
        */
        thrust::device_vector<float> Up_dev_;

        //! A private member variable.
        /*!
            全エネルギー
        */
        T Utot_;

        //! A private member variable.
        /*!
            n個目の原子の速度（デバイス側）
        */
        thrust::device_vector<float4> V_dev_;

        //! A private member variable (constant).
        /*!
            ポテンシャルエネルギーの打ち切り
        */
        T const Vrc_;
        
        // #endregion メンバ変数

        // #region 禁止されたコンストラクタ・メンバ関数

        //! A private copy constructor (deleted).
        /*!
            コピーコンストラクタ（禁止）
        */
        Ar_moleculardynamics(Ar_moleculardynamics const &) = delete;

        //! A private member function (deleted).
        /*!
            operator=()の宣言（禁止）
            \param コピー元のオブジェクト（未使用）
            \return コピー元のオブジェクト
        */
        Ar_moleculardynamics & operator=(Ar_moleculardynamics const &) = delete;

        // #endregion 禁止されたコンストラクタ・メンバ関数
    };

    // #region static private 定数

    template <typename T>
    T const Ar_moleculardynamics<T>::FIRSTSCALE = 1.0;
    
    template <typename T>
    T const Ar_moleculardynamics<T>::FIRSTTEMP = 50.0;

    template <typename T>
    T const Ar_moleculardynamics<T>::ALPHA = 0.2;

    template <typename T>
    T const Ar_moleculardynamics<T>::DT = 0.001;

    template <typename T>
    T const Ar_moleculardynamics<T>::KB = 1.3806488E-23;

    template <typename T>
    T const Ar_moleculardynamics<T>::YPSILON = 1.6540172624E-21;

    // #endregion static private 定数

    // #region コンストラクタ

    template <typename T>
    Ar_moleculardynamics<T>::Ar_moleculardynamics()
        :
        dt2(DT * DT),
        F_dev_(Nc_ * Nc_ * Nc_ * 4),
        rc2_(rc_ * rc_),
        rcm6_(std::pow(rc_, -6.0)),
        rcm12_(std::pow(rc_, -12.0)),
        Tg_(Ar_moleculardynamics::FIRSTTEMP * Ar_moleculardynamics::KB / Ar_moleculardynamics::YPSILON),
        r_dev_(Nc_ * Nc_ * Nc_ * 4),
        r1_dev_(Nc_ * Nc_ * Nc_ * 4),
        Up_dev_(Nc_ * Nc_ * Nc_ * 4),
        V_dev_(Nc_ * Nc_ * Nc_ * 4),
        Vrc_(4.0 * (rcm12_ - rcm6_))
    {
        // initalize parameters
        lat_ = std::pow(2.0, 2.0 / 3.0) * scale_;

        MD_iter_ = 1;

        MD_initPos();
        MD_initVel();

        periodiclen_ = lat_ * static_cast<T>(Nc_);
    }

    // #endregion コンストラクタ

    // #region publicメンバ関数

    template <typename T>
    void Ar_moleculardynamics<T>::Calc_Forces()
    {
        // 各原子に働く力の初期化
        thrust::fill(thrust::device, F_dev_.begin(), F_dev_.end(), float4());
        
        // 各原子に働くポテンシャルエネルギーの初期化
        thrust::fill(thrust::device, Up_dev_.begin(), Up_dev_.end(), 0.0f);

        // 力の計算
        Calc_Forces_Core <<<Ar_moleculardynamics::SMNUM, NumAtom_ / Ar_moleculardynamics::SMNUM>>>(
            thrust::raw_pointer_cast(r_dev_.data()),
            thrust::raw_pointer_cast(F_dev_.data()),
            thrust::raw_pointer_cast(Up_dev_.data()),
            ncp_,
            NumAtom_,
            periodiclen_,
            rc2_,
            Vrc_);
        
        // ポテンシャルエネルギーの計算
        Up_ = thrust::reduce(thrust::device, Up_dev_.begin(), Up_dev_.end(), 0.0f);
    }

    template <typename T>
    void Ar_moleculardynamics<T>::Move_Atoms()
    {
        // 運動エネルギーの計算
        thrust::device_vector<float> tmp(NumAtom_);
        thrust::transform(V_dev_.begin(), V_dev_.end(), tmp.begin(), norm2());
        Uk_ = thrust::reduce(thrust::device, tmp.begin(), tmp.end(), 0.0f) * 0.5;

        // 全エネルギー（運動エネルギー+ポテンシャルエネルギー）の計算
        Utot_ = Uk_ + Up_;

        std::cout << boost::format("MDステップ = %d, ポテンシャル = %.8f, 運動エネルギー = %.8f\n") % MD_iter_ % Up_ % Uk_;

        // 温度の計算
        Tc_ = Uk_ / (1.5 * static_cast<T>(NumAtom_));

        // calculate temperture
        auto const s = std::sqrt((Tg_ + Ar_moleculardynamics::ALPHA * (Tc_ - Tg_)) / Tc_);

        switch (MD_iter_) {
        case 1:
            Move_Atoms1_Core<<<Ar_moleculardynamics::SMNUM, NumAtom_ / Ar_moleculardynamics::SMNUM >>>(
                thrust::raw_pointer_cast(F_dev_.data()),
                thrust::raw_pointer_cast(r_dev_.data()),
                thrust::raw_pointer_cast(r1_dev_.data()),
                thrust::raw_pointer_cast(V_dev_.data()),
                Ar_moleculardynamics::DT,
                s);
        break;

        default:
            Move_Atoms_Core<<<Ar_moleculardynamics::SMNUM, NumAtom_ / Ar_moleculardynamics::SMNUM>>>(
                thrust::raw_pointer_cast(F_dev_.data()),
                thrust::raw_pointer_cast(r_dev_.data()),
                thrust::raw_pointer_cast(r1_dev_.data()),
                thrust::raw_pointer_cast(V_dev_.data()),
                Ar_moleculardynamics::DT,
                s);
        break;
        }

        check_periodic<<<Ar_moleculardynamics::SMNUM, NumAtom_ / Ar_moleculardynamics::SMNUM >>>(
            thrust::raw_pointer_cast(r_dev_.data()),
            thrust::raw_pointer_cast(r1_dev_.data()),
            periodiclen_);

        MD_iter_++;
    }

    // #endregion publicメンバ関数

    // #region privateメンバ関数

    template <typename T>
    void Ar_moleculardynamics<T>::MD_initPos()
    {
        T sx, sy, sz;
        auto n = 0;
        thrust::host_vector<float4> r(Nc_ * Nc_ * Nc_ * 4);

        for (auto i = 0; i < Nc_; i++) {
            for (auto j = 0; j < Nc_; j++) {
                for (auto k = 0; k < Nc_; k++) {
                    // 基本セルをコピーする
                    sx = static_cast<T>(i) * lat_;
                    sy = static_cast<T>(j) * lat_;
                    sz = static_cast<T>(k) * lat_;

                    // 基本セル内には4つの原子がある
                    r[n].x = sx;
                    r[n].y = sy;
                    r[n].z = sz;
                    n++;

                    r[n].x = 0.5 * lat_ + sx;
                    r[n].y = 0.5 * lat_ + sy;
                    r[n].z = sz;
                    n++;

                    r[n].x = sx;
                    r[n].y = 0.5 * lat_ + sy;
                    r[n].z = 0.5 * lat_ + sz;
                    n++;

                    r[n].x = 0.5 * lat_ + sx;
                    r[n].y = sy;
                    r[n].z = 0.5 * lat_ + sz;
                    n++;
                }
            }
        }

        NumAtom_ = n;

        // move the center of mass to the origin
        // 系の重心を座標系の原点とする
        sx = 0.0;
        sy = 0.0;
        sz = 0.0;

        for (auto n = 0; n < NumAtom_; n++) {
            sx += r[n].x;
            sy += r[n].y;
            sz += r[n].z;
        }

        sx /= static_cast<T>(NumAtom_);
        sy /= static_cast<T>(NumAtom_);
        sz /= static_cast<T>(NumAtom_);

        for (auto n = 0; n < NumAtom_; n++) {
            r[n].x -= sx;
            r[n].y -= sy;
            r[n].z -= sz;
        }

        // ホスト→デバイス
        thrust::copy(r.begin(), r.end(), r_dev_.begin());
    }

    template <typename T>
    void Ar_moleculardynamics<T>::MD_initVel()
    {
        auto const v = std::sqrt(3.0 * Tg_);

        myrandom::MyRand mr(-1.0, 1.0);

        auto const generator4 = [this, &mr, v]() {
            norm2 const n2;
            auto rnd = make_float4(mr.myrand(), mr.myrand(), mr.myrand(), 0.0f);
            auto const tmp = 1.0 / std::sqrt(n2(rnd));
            rnd.x *= tmp;
            rnd.y *= tmp;
            rnd.z *= tmp;
            
            // 方向はランダムに与える
            return make_float4(v * rnd.x, v * rnd.y, v * rnd.z, 0.0f);
        };

        thrust::host_vector<float4> V(NumAtom_);
        boost::generate(V, generator4);

        auto sx = 0.0;
        auto sy = 0.0;
        auto sz = 0.0;

        for (auto n = 0; n < NumAtom_; n++) {
            sx += V[n].x;
            sy += V[n].y;
            sz += V[n].z;
        }

        sx /= static_cast<T>(NumAtom_);
        sy /= static_cast<T>(NumAtom_);
        sz /= static_cast<T>(NumAtom_);

        // 重心の並進運動を避けるために、速度の和がゼロになるように補正
        for (auto n = 0; n < NumAtom_; n++) {
            V[n].x -= sx;
            V[n].y -= sy;
            V[n].z -= sz;
        }

        // ホスト→デバイス
        thrust::copy(V.begin(), V.end(), V_dev_.begin());
    }

    // #endregion privateメンバ関数

    // #region 非メンバ関数

    __global__ void Calc_Forces_Core(
        float4 const * rv,
        float4 * F,
        float * Up,
        std::int32_t const ncp,
        std::int32_t const numatom,
        float const periodiclen,
        float const rc2,
        float const Vrc)
    {
        auto const n = blockIdx.x * blockDim.x + threadIdx.x;

        for (auto m = 0; m < numatom; m++) {

            // ±ncp分のセル内の原子との相互作用を計算
            for (auto i = -ncp; i <= ncp; i++) {
                for (auto j = -ncp; j <= ncp; j++) {
                    for (auto k = -ncp; k <= ncp; k++) {
                        auto s = make_float4(
                            static_cast<float>(i) * periodiclen,
                            static_cast<float>(j) * periodiclen,
                            static_cast<float>(k) * periodiclen,
                            0.0f);

                        // 自分自身との相互作用を排除
                        if (n != m || i != 0 || j != 0 || k != 0) {
                            auto const rn = rv[n];
                            auto const rm = rv[m];
                            auto const dx = rn.x - (rm.x + s.x);
                            auto const dy = rn.y - (rm.y + s.y);
                            auto const dz = rn.z - (rm.z + s.z);
                            
                            auto const r2 = dx * dx + dy * dy + dz * dz;
                            // 打ち切り距離内であれば計算
                            if (r2 <= rc2) {
                                auto const r = std::sqrt(r2);
                                auto const rm6 = 1.0 / (r2 * r2 * r2);
                                auto const rm7 = rm6 / r;
                                auto const rm12 = rm6 * rm6;
                                auto const rm13 = rm12 / r;

                                auto const Fr = 48.0 * rm13 - 24.0 * rm7;
                                
                                F[n].x += dx / r * Fr;
                                F[n].y += dy / r * Fr;
                                F[n].z += dz / r * Fr;
                                Up[n] += 0.5 * (4.0 * (rm12 - rm6) - Vrc);
                            }
                        }
                    }
                }
            }
        }
    }

    __global__ void check_periodic(
        float4 * r,
        float4 * r1,
        float const periodiclen)
    {
        // consider the periodic boundary condination
        // セルの外側に出たら座標をセル内に戻す
        auto const n = blockIdx.x * blockDim.x + threadIdx.x;

        if (r[n].x > periodiclen) {
            r[n].x -= periodiclen;
            r1[n].x -= periodiclen;
        }
        else if (r[n].x < 0.0f) {
            r[n].x += periodiclen;
            r1[n].x += periodiclen;
        }

        if (r[n].y > periodiclen) {
            r[n].y -= periodiclen;
            r1[n].y -= periodiclen;
        }
        else if (r[n].y < 0.0f) {
            r[n].y += periodiclen;
            r1[n].y += periodiclen;
        }

        if (r[n].z > periodiclen) {
            r[n].z -= periodiclen;
            r1[n].z -= periodiclen;
        }
        else if (r[n].z < 0.0f) {
            r[n].z += periodiclen;
            r1[n].z += periodiclen;
        }
    }

    __global__ void Move_Atoms1_Core(
        float4 const * F,
        float4 * r,
        float4 * r1,
        float4 * V,
        float const dt,
        float const s)
    {
        // update the coordinates by the second order Euler method
        // 最初のステップだけ修正Euler法で時間発展
        auto const n = blockIdx.x * blockDim.x + threadIdx.x;
        auto const dt2 = dt * dt;

        r1[n].x = r[n].x;
        r1[n].y = r[n].y;
        r1[n].z = r[n].z;

        // scaling of velocity
        V[n].x *= s;
        V[n].y *= s;
        V[n].z *= s;

        // update coordinates and velocity
        r[n].x += dt * V[n].x + 0.5 * F[n].x * dt2;
        r[n].y += dt * V[n].y + 0.5 * F[n].y * dt2;
        r[n].z += dt * V[n].z + 0.5 * F[n].z * dt2;

        V[n].x += dt * F[n].x;
        V[n].y += dt * F[n].y;
        V[n].z += dt * F[n].z;
    }
    
    __global__ void Move_Atoms_Core(
        float4 const * F,
        float4 * r,
        float4 * r1,
        float4 * V,
        float const dt,
        float const s)
    {
        // update the coordinates by the Verlet method
        auto const n = blockIdx.x * blockDim.x + threadIdx.x;

        auto const dt2 = dt * dt;
        auto const rtmp = r[n];

        // update coordinates and velocity
        // Verlet法の座標更新式において速度成分を抜き出し、その部分をスケールする
        r[n].x += s * (r[n].x - r1[n].x) + F[n].x * dt2;
        r[n].y += s * (r[n].y - r1[n].y) + F[n].y * dt2;
        r[n].z += s * (r[n].z - r1[n].z) + F[n].z * dt2;

        V[n].x = 0.5f * (r[n].x - r1[n].x) / dt;
        V[n].y = 0.5f * (r[n].y - r1[n].y) / dt;
        V[n].z = 0.5f * (r[n].z - r1[n].z) / dt;

        r1[n] = rtmp;
    }

    // #endregion 非メンバ関数
}

#endif      // _AR_MOLECULARDYNAMICS_H_
