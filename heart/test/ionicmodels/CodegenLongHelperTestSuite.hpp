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

/*
CellMl files are found using cmake_fetch in /heart/test/CMakeLists.txt
the CellML files will be downloaded on the cmake step into _deps/cellml_repo-src/cellml/
*/


#ifndef CODEGENLONGHELPERCLASS_HPP_
#define CODEGENLONGHELPERCLASS_HPP_

#include <cxxtest/TestSuite.h>

#include <algorithm>
#include <boost/assign/list_of.hpp>
#include <boost/foreach.hpp>
#include <cstring>
#include <iostream>
#include <vector>
#include <cmath>

#include "AbstractCardiacCell.hpp"
#include "AbstractCardiacCellInterface.hpp"
#include "AbstractCvodeCell.hpp"
#include "AbstractGeneralizedRushLarsenCardiacCell.hpp"
#include "AbstractRushLarsenCardiacCell.hpp"
#include "Timer.hpp"

#include "DynamicLoadingHelperFunctions.hpp"

#include "CellMLToSharedLibraryConverter.hpp"
#include "DynamicCellModelLoader.hpp"
#include "DynamicModelLoaderRegistry.hpp"
#include "FileFinder.hpp"

#include "CellProperties.hpp"
#include "Exception.hpp"
#include "HeartConfig.hpp"
#include "RunAndCheckIonicModels.hpp"
#include "Warnings.hpp"

/**
 * Helper class to allow us to split the CodegenLong tests into multiple test suites.
 */
class CodegenLongHelperTestSuite : public CxxTest::TestSuite
{

private:
    bool mUseCvodeJacobian = true;
    double cvOdeTolerances = DOUBLE_UNSET;

    double GetAttribute(boost::shared_ptr<AbstractCardiacCellInterface> pCell,
                        const std::string& rAttrName,
                        double defaultValue)
    {
        AbstractUntemplatedParameterisedSystem* p_system
            = dynamic_cast<AbstractUntemplatedParameterisedSystem*>(pCell.get());
        assert(p_system);
        double attr_value;
        if (p_system->HasAttribute(rAttrName))
        {
            attr_value = p_system->GetAttribute(rAttrName);
        }
        else
        {
            attr_value = defaultValue;
        }
        return attr_value;
    }

    void Simulate(const std::string& rOutputDirName,
                  const std::string& rModelName,
                  boost::shared_ptr<AbstractCardiacCellInterface> pCell)
    {
        double end_time = GetAttribute(pCell, "SuggestedCycleLength", 700.0); // ms
        if (pCell->GetSolver() || dynamic_cast<AbstractRushLarsenCardiacCell*>(pCell.get()))
        {
            double dt = GetAttribute(pCell, "SuggestedForwardEulerTimestep", 0.0);
            if (dt > 0.0)
            {
                pCell->SetTimestep(dt);
            }
        }
#ifdef CHASTE_CVODE
        AbstractCvodeSystem* p_cvode_cell = dynamic_cast<AbstractCvodeSystem*>(pCell.get());
        if (p_cvode_cell)
        {
            // Set tolerances
            if (cvOdeTolerances != DOUBLE_UNSET)
            {
                p_cvode_cell->SetTolerances(cvOdeTolerances, cvOdeTolerances);
            }
            // Set a larger max internal time steps per sampling interval (CVODE's default is 500)
            p_cvode_cell->SetMaxSteps(1000);
            // Numerical or analytic J for CVODE?
            if (!mUseCvodeJacobian)
            {
                p_cvode_cell->ForceUseOfNumericalJacobian();
            }
        }
#endif
        double sampling_interval = 1.0; // ms; used as max dt for CVODE too
        Timer::Reset();
        OdeSolution solution = pCell->Compute(0.0, end_time, sampling_interval);
        std::stringstream message;
        message << "Model " << rModelName << " writing to " << rOutputDirName << " took";
        Timer::Print(message.str());

        const unsigned output_freq = 10; // Only output every N samples
        solution.WriteToFile(rOutputDirName, rModelName, "ms", output_freq, false);
        // Check an AP was produced
        std::vector<double> voltages = solution.GetVariableAtIndex(pCell->GetVoltageIndex());
        CellProperties props(voltages, solution.rGetTimes());
        props.GetLastActionPotentialDuration(90.0); // Don't catch the exception here if it's thrown
        // Compare against saved results
        CheckResults(rModelName, voltages, solution.rGetTimes(), output_freq);
    }

    void CheckResults(const std::string& rModelName,
                      std::vector<double>& rVoltages,
                      std::vector<double>& rTimes,
                      unsigned outputFreq,
                      double tolerance = 1.0)
    {
        // Read data entries for the reference file
        ColumnDataReader data_reader("heart/test/data/cellml", rModelName, false);
        std::vector<double> valid_times = data_reader.GetValues("Time");
        std::vector<double> valid_voltages = GetVoltages(data_reader);

        TS_ASSERT_EQUALS(rTimes.size(), (valid_times.size() - 1) * outputFreq + 1);
        for (unsigned i = 0; i < valid_times.size(); i++)
        {
            TS_ASSERT_DELTA(rTimes[i * outputFreq], valid_times[i], 1e-12);
            TS_ASSERT_DELTA(rVoltages[i * outputFreq], valid_voltages[i], tolerance);
        }
    }

    void RunTest(const std::string& rOutputDirName,
                 const std::string& rModelName,
                 const std::vector<std::string>& rArgs,
                 bool testLookupTables = false,
                 double tableTestV = -1000)
    {
        // Copy CellML file into output dir
        OutputFileHandler handler(rOutputDirName, true);
        FileFinder cellml_file("_deps/cellml_repo-src/cellml/" + rModelName + ".cellml", RelativeTo::ChasteBuildRoot);
        handler.CopyFileTo(cellml_file);

        //Out files no longer needed with codegen
        // Set options (config files no longer used)
        std::vector<std::string> args(rArgs);
        //        args.push_back("--profile");
        CellMLToSharedLibraryConverter converter(true);
        converter.SetOptions(rArgs);

        // Do the conversion
        FileFinder copied_file(rOutputDirName + "/" + rModelName + ".cellml", RelativeTo::ChasteTestOutput);
        DynamicCellModelLoaderPtr p_loader = converter.Convert(copied_file);
        // Apply a stimulus of -40 uA/cm^2 - should work for all models
        boost::shared_ptr<AbstractCardiacCellInterface> p_cell(CreateCellWithStandardStimulus(*p_loader, -40.0));

        // Check that the default stimulus units are correct
        if (p_cell->HasCellMLDefaultStimulus())
        {
            // Record the existing stimulus and re-apply it at the end
            boost::shared_ptr<AbstractStimulusFunction> original_stim = p_cell->GetStimulusFunction();

            // Tell the cell to use the default stimulus and retrieve it
            boost::shared_ptr<RegularStimulus> p_reg_stim = p_cell->UseCellMLDefaultStimulus();

            if (rModelName != "aslanidi_Purkinje_model_20099") // Even before recent changes aslanidi model has stimulus of -400 !
            {
                // Stimulus magnitude should be approximately between -5 and -81 uA/cm^2
                TS_ASSERT_LESS_THAN(p_reg_stim->GetMagnitude(), -5);
                TS_ASSERT_LESS_THAN(-81, p_reg_stim->GetMagnitude());
            }

            // Stimulus duration should be approximately between 0.1 and 5 ms.
            TS_ASSERT_LESS_THAN(p_reg_stim->GetDuration(), 6.01);
            TS_ASSERT_LESS_THAN(0.1, p_reg_stim->GetDuration());

            // Stimulus period should be approximately between 70 (for bondarenko - seems fast! - would expect 8-10 beats per second for mouse) and 2000ms.
            TS_ASSERT_LESS_THAN(p_reg_stim->GetPeriod(), 2000);
            TS_ASSERT_LESS_THAN(70, p_reg_stim->GetPeriod());

            p_cell->SetIntracellularStimulusFunction(original_stim);
        }

        // Check lookup tables exist if they should and throw appropriate errors if we go outside their range...
        if (testLookupTables && rModelName != "hodgkin_huxley_squid_axon_model_1952_modified")
        {
            double v = p_cell->GetVoltage();
            p_cell->SetVoltage(tableTestV);

            // This try catch used to be TS_ASSERT_THROWS_CONTAINS (see #2982)
            try
            {
                p_cell->GetIIonic();
            }
            catch (const Exception& e)
            {
                TS_ASSERT(e.CheckShortMessageContains("outside lookup table range") == "");
            }

            p_cell->SetVoltage(v);
        }
        std::cout << "Running simulation...\n"
                  << std::flush;
        Simulate(rOutputDirName, rModelName, p_cell);
    }

protected:
    std::vector<std::string> models;
    std::vector<std::string> easy_models;

    // Returns a list of all models in special_treatment_models that exist in all_models and deletes these models from all_models
    std::vector<std::string> special_treatment_models(std::vector<std::string>& all_models, std::vector<std::string> special_treatment_models)
    {
        BOOST_FOREACH (std::string model, special_treatment_models)
        {
            auto special_model = std::find(all_models.begin(), all_models.end(), model);
            bool found = special_model != all_models.end();
            if(found)
            {
                all_models.erase(special_model);
            }
            else
            {
                special_treatment_models.erase(std::find(special_treatment_models.begin(), special_treatment_models.end(), model));

            }
        }
        return special_treatment_models;
    }

    void setUp()
    {
        AddAllModels(models);
        AddEasyModels(easy_models);
    }

public:
    void RunTests(const std::string& rOutputDirName,
                  const std::vector<std::string>& rModels,
                  const std::vector<std::string>& rArgs,
                  bool testLookupTables = false,
                  double tableTestV = -1000,
                  bool warningsOk = true)
    {
        OutputFileHandler handler(rOutputDirName); // Clear folder (collective)
        PetscTools::IsolateProcesses(true); // Simple parallelism
        std::vector<std::string> failures;
        for (unsigned i = 0; i < rModels.size(); ++i)
        {
            if (PetscTools::IsParallel() && i % PetscTools::GetNumProcs() != PetscTools::GetMyRank())
            {
                continue; // Let someone else do this model
            }
            try
            {
                unsigned num_failed_asserts = CxxTest::tracker().testFailedAsserts();
                RunTest(rOutputDirName + "/" + rModels[i], rModels[i], rArgs, testLookupTables, tableTestV);
                if (CxxTest::tracker().testFailedAsserts() > num_failed_asserts)
                {
                    std::cout << "Counted a failed TS_ASSERT\n"
                              << std::flush;
                    EXCEPTION((CxxTest::tracker().testFailedAsserts() - num_failed_asserts) << " test assertion failure(s).");
                }
            }
            catch (const Exception& e)
            {
                failures.push_back(rModels[i]);
                TS_FAIL("Failure testing cell model " + rModels[i] + ": " + e.GetMessage());
            }
            if (!warningsOk)
            {
                if (rModels[i] == "demir_model_1994" || strncmp((rModels[i]).c_str(), "zhang_SAN_model_2000", strlen("zhang_SAN_model_2000")) == 0)
                {
                    // We know this model does something that provokes one warning...
                    TS_ASSERT_EQUALS(Warnings::Instance()->GetNumWarnings(), 1u);
                }
                else if(rModels[i] == "courtemanche_ramirez_nattel_1998")
                {
                    // Can throw 1 lookup table warning
                   TS_ASSERT_LESS_THAN(Warnings::Instance()->GetNumWarnings(), 2u)
                }
                else
                {
                    TS_ASSERT_EQUALS(Warnings::Instance()->GetNumWarnings(), 0u);
                }
            }
            Warnings::NoisyDestroy(); // Print out any warnings now, not at program exit
        }
        // Wait for all simulations to finish before printing summary of failures
        PetscTools::IsolateProcesses(false);
        PetscTools::Barrier("RunTests");

        if (!failures.empty())
        {
            std::cout << failures.size() << " models failed for " << rOutputDirName << ":" << std::endl;
            for (unsigned i = 0; i < failures.size(); ++i)
            {
                std::cout << "   " << failures[i] << std::endl;
            }
        }
    }

    void AddAllModels(std::vector<std::string>& rModels)
    {
        rModels.emplace_back("aslanidi_Purkinje_model_2009");
        rModels.emplace_back("courtemanche_ramirez_nattel_1998");
        rModels.emplace_back("decker_2009");
        rModels.emplace_back("demir_model_1994");
        rModels.emplace_back("dokos_model_1996");
        rModels.emplace_back("fink_noble_giles_model_2008");
        rModels.emplace_back("grandi_pasqualini_bers_2010_ss");
        rModels.emplace_back("hund_rudy_2004");
        rModels.emplace_back("iyer_2004");
        rModels.emplace_back("iyer_model_2007");
        rModels.emplace_back("jafri_rice_winslow_model_1998");
        rModels.emplace_back("livshitz_rudy_2007");
        rModels.emplace_back("luo_rudy_1994");
        rModels.emplace_back("noble_model_1991");
        rModels.emplace_back("noble_model_1998");
        rModels.emplace_back("nygren_atrial_model_1998");
        rModels.emplace_back("pandit_clark_giles_demir_2001_epicardial_cell");
        rModels.emplace_back("priebe_beuckelmann_1998");
        rModels.emplace_back("shannon_wang_puglisi_weber_bers_2004");
        rModels.emplace_back("stewart_zhang_model_2008_ss");
        rModels.emplace_back("ten_tusscher_model_2004_endo");
        rModels.emplace_back("ten_tusscher_model_2004_epi");
        rModels.emplace_back("ten_tusscher_model_2006_epi");
        rModels.emplace_back("viswanathan_model_1999_epi");
        rModels.emplace_back("winslow_model_1999");

        // Additional models added for testing when introducing codegen, as these caused some unit conversion isssues (in ApPredict)
        rModels.emplace_back("difrancesco_noble_model_1985");
        rModels.emplace_back("faber_rudy_2000");
        rModels.emplace_back("li_mouse_2010");
        rModels.emplace_back("paci_hyttinen_aaltosetala_severi_ventricularVersion");
        rModels.emplace_back("sachse_moreno_abildskov_2008_b");
    }

    // These models passed first time requiring no special treatment for any model type
    void AddEasyModels(std::vector<std::string>& rModels)
    {
       rModels.emplace_back("beeler_reuter_model_1977");
       rModels.emplace_back("bondarenko_szigeti_bett_kim_rasmusson_2004_apical");
       rModels.emplace_back("earm_noble_model_1990");
       rModels.emplace_back("espinosa_model_1998_normal");
       rModels.emplace_back("hilgemann_noble_model_1987");
       rModels.emplace_back("hodgkin_huxley_squid_axon_model_1952_modified");
       rModels.emplace_back("iribe_model_2006");
       rModels.emplace_back("kurata_model_2002");
       rModels.emplace_back("mahajan_shiferaw_2008");
       rModels.emplace_back("matsuoka_model_2003");
       rModels.emplace_back("noble_noble_SAN_model_1984");
       rModels.emplace_back("noble_SAN_model_1989");
       rModels.emplace_back("sakmann_model_2000_epi");
       rModels.emplace_back("zhang_SAN_model_2000_0D_capable");
    }


    void SetUseCvodeJacobian(bool useCvodeJacobian)
    {
        mUseCvodeJacobian = useCvodeJacobian;
    }

    void SetUseCvOdeTolerances(double tolerances)
    {
        cvOdeTolerances = tolerances;
    }
};

#endif // CODEGENLONGHELPERCLASS_HPP_
