/*

Copyright (c) 2005-2024, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef WNTCELLCYCLEODESYSTEM_HPP_
#define WNTCELLCYCLEODESYSTEM_HPP_

#include "ChasteSerialization.hpp"
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <cmath>
#include <iostream>

#include "AbstractOdeSystem.hpp"
#include "AbstractCellMutationState.hpp"

/**
 * Represents the Mirams et al. system of ODEs, based on Swat et al. (2004)
 * [doi:10.1093/bioinformatics/bth110]
 * and a simple Wnt model (unpublished)
 *
 * The variables are
 *
 * 0. r = pRb
 * 1. e = E2F1
 * 2. i = CycD (inactive)
 * 3. j = CycD (active)
 * 4. p = pRb-p
 * 5. c = destruction complex (Active)
 * 6. b1 = Beta-Catenin (from 1st allele)
 * 7. b2 = Beta-Catenin (from 1st allele)
 * 8. WntLevel
 */
class WntCellCycleOdeSystem : public AbstractOdeSystem
{
private:

    /**
     * Parameters for the Swat et al. (2004) model
     */

    /** Dimensional parameter k_2. */
    double mk2d;
    /** Dimensional parameter k_3. */
    double mk3d;
    /** Dimensional parameter k_34. */
    double mk34d;
    /** Dimensional parameter k_2. */
    double mk43d;
    /** Dimensional parameter k_23. */
    double mk23d;
    /** Dimensional parameter a. */
    double mad;
    /** Dimensional parameter J_11. */
    double mJ11d;
    /** Dimensional parameter J_12. */
    double mJ12d;
    /** Dimensional parameter J_13. */
    double mJ13d;
    /** Dimensional parameter J_13. */
    double mJ61d;
    /** Dimensional parameter J_62. */
    double mJ62d;
    /** Dimensional parameter J_63. */
    double mJ63d;
    /** Dimensional parameter K_m1. */
    double mKm1d;
    /** Dimensional parameter k_p. */
    double mkpd;
    /** Dimensionless parameter phi_r. */
    double mphi_r;
    /** Dimensionless parameter phi_i. */
    double mphi_i;
    /** Dimensionless parameter phi_j. */
    double mphi_j;
    /** Dimensionless parameter phi_p. */
    double mphi_p;
    /** Dimensional parameter a_2. */
    double ma2d;
    /** Dimensional parameter a_3. */
    double ma3d;
    /** Dimensional parameter a_4. */
    double ma4d;
    /** Dimensional parameter a_5. */
    double ma5d;
    /** Dimensional parameter k_16. */
    double mk16d;
    /** Dimensional parameter k_61. */
    double mk61d;
    /** Dimensionless parameter phi_E2F1. */
    double mPhiE2F1;

    /** The mutation state of the cell - Wnt pathway behaviour (and hence cell cycle time) changes depending on this */
    boost::shared_ptr<AbstractCellMutationState> mpMutationState;

    /** The Wnt level (this affects the ODE system). */
    double mWntLevel;

    friend class boost::serialization::access;
    /**
     * Serialize the object and its member variables.
     *
     * @param archive the archive
     * @param version the current version of this class
     */
    template<class Archive>
    void serialize(Archive & archive, const unsigned int version)
    {
        archive & boost::serialization::base_object<AbstractOdeSystem>(*this);
    }

public:

    /**
     * Constructor.
     *
     * @param wntLevel is a non-dimensional Wnt value between 0 and 1. This sets up the Wnt pathway in its steady state.
     * @param pMutationState optional mutation state (affects the ODE system)
     * @param stateVariables optional initial conditions for state variables (only used in archiving)
     */
    WntCellCycleOdeSystem(double wntLevel=0.0,
                          boost::shared_ptr<AbstractCellMutationState> pMutationState=boost::shared_ptr<AbstractCellMutationState>(),
                          std::vector<double> stateVariables=std::vector<double>());

    /**
     * Destructor.
     */
    ~WntCellCycleOdeSystem();

    /**
     * Initialise parameter values.
     */
    void Init();

    /**
     * Set the mutation state of the cell.
     *
     * This should be called by the relevant cell-cycle model before any solving
     * of the ODE system (as it is used to evaluate the Y derivatives).
     *
     * @param pMutationState the mutation state.
     */
    void SetMutationState(boost::shared_ptr<AbstractCellMutationState> pMutationState);

    /**
     * Called by the archive function on the Wnt cell-cycle model.
     *
     * @return #mpMutationState the mutation state of the cell.
     */
    const boost::shared_ptr<AbstractCellMutationState> GetMutationState() const;

    /**
     * Compute the RHS of the WntCellCycle system of ODEs.
     *
     * Returns a vector representing the RHS of the ODEs at each time step, y' = [y1' ... yn'].
     * An ODE solver will call this function repeatedly to solve for y = [y1 ... yn].
     *
     * @param time used to evaluate the RHS.
     * @param rY value of the solution vector used to evaluate the RHS.
     * @param rDY filled in with the resulting derivatives (using Alarcons et al. (2004) system of equations).
     */
    void EvaluateYDerivatives(double time, const std::vector<double>& rY, std::vector<double>& rDY);

    /**
     * This also contains a calculation of dY[1], copied from EvaluateYDerivatives.
     * Ensure they do not get out of sync!
     *
     * @param time at which to calculate whether the stopping event has occurred
     * @param rY value of the solution vector used to evaluate the RHS
     *
     * @return whether we have reached the stopping event
     */
    bool CalculateStoppingEvent(double time, const std::vector<double>& rY);

    /**
     * When using CVODE this function is called instead of CalculateStoppingEvent.
     * It allows the point at which rY[1] reaches 1 to be found to greater precision.
     *
     * @param time at which to calculate whether the stopping event has occurred
     * @param rY value of the solution vector used to evaluate the RHS
     *
     * @return function value - giving CVODE an estimate of how close we are to the root.
     */
    double CalculateRootFunction(double time, const std::vector<double>& rY);

    /**
     * @return #mWntLevel.
     */
    double GetWntLevel() const;
};

// Declare identifier for the serializer
#include "SerializationExportWrapper.hpp"
CHASTE_CLASS_EXPORT(WntCellCycleOdeSystem)

namespace boost
{
namespace serialization
{
/**
 * Serialize information required to construct a WntCellCycleOdeSystem.
 */
template<class Archive>
inline void save_construct_data(
    Archive & ar, const WntCellCycleOdeSystem * t, const unsigned int file_version)
{
    // Save data required to construct instance
    const double wnt_level = t->GetWntLevel();
    ar & wnt_level;

    const boost::shared_ptr<AbstractCellMutationState> p_mutation_state = t->GetMutationState();
    ar & p_mutation_state;

    const std::vector<double>& state_variables = t->rGetConstStateVariables();
    ar & state_variables;
}

/**
 * De-serialize constructor parameters and initialise a WntCellCycleOdeSystem.
 */
template<class Archive>
inline void load_construct_data(
    Archive & ar, WntCellCycleOdeSystem * t, const unsigned int file_version)
{
    // Retrieve data from archive required to construct new instance
    double wnt_level;
    ar & wnt_level;

    boost::shared_ptr<AbstractCellMutationState> p_mutation_state;
    ar & p_mutation_state;

    std::vector<double> state_variables;
    ar & state_variables;

    // Invoke inplace constructor to initialise instance
    ::new(t)WntCellCycleOdeSystem(wnt_level, p_mutation_state, state_variables);
}
}
} // namespace ...

#endif /*WNTCELLCYCLEODESYSTEM_HPP_*/
