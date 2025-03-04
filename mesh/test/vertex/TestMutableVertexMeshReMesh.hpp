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

#ifndef TESTMUTABLEVERTEXMESHREMESH_HPP_
#define TESTMUTABLEVERTEXMESHREMESH_HPP_

#include <cxxtest/TestSuite.h>

#include "FileComparison.hpp"
#include "MutableVertexMesh.hpp"
#include "VertexMeshWriter.hpp"
#include "Warnings.hpp"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "ArchiveOpener.hpp"

//This test is always run sequentially (never in parallel)
#include "FakePetscSetup.hpp"

/*
 * Testing of edge infrastructure is also done here.
 * In order to avoid duplication, tests for some edge infrastructure are also implemented in
 * TestMutableVerteMeshOperationsWithPopulationSrn. There, explicit tests for correct edge rearrangements with SRNs
 * are implemented.
 */
class TestMutableVertexMeshReMesh : public CxxTest::TestSuite
{
public:
    void TestPerformNodeMerge()
    {
        /*
         * Create a mesh comprising a single triangular element, as shown below.
         * We will test that the nodes marked with an x are merged correctly.
         *
         *      /|
         *     / |
         *    /  |
         *   /   |
         *  /    |
         *  --xx-
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.4, 0.0));
        nodes.push_back(new Node<2>(4, true, 0.6, 0.0));

        unsigned node_indices_elem_0[5] = { 0, 3, 4, 1, 2 };
        std::vector<Node<2>*> nodes_elem_0;
        for (unsigned i = 0; i < 5; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        //For testing edge operations
        vertex_mesh.SetMeshOperationTracking(true);

        // Merge nodes 3 and 4
        vertex_mesh.IdentifySwapType(vertex_mesh.GetNode(3), vertex_mesh.GetNode(4));

        // Test if the node merge operation has been recorded properly
        auto operation_recorder = vertex_mesh.GetOperationRecorder();
        const std::vector<EdgeOperation>& edge_operations = operation_recorder->GetEdgeOperations();
        const unsigned num_operations = edge_operations.size();
        TS_ASSERT_EQUALS(num_operations, 1u);
        TS_ASSERT_EQUALS(edge_operations[0].GetOperation(), EDGE_OPERATION_NODE_MERGE);

        // Test the mesh is correctly updated
        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 1u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 4u);

        // Test the correct nodes are boundary nodes
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetNode(i)->IsBoundaryNode(), true);
        }

        // Test the merged node is in the correct place
        TS_ASSERT_DELTA(vertex_mesh.GetNode(3)->rGetLocation()[0], 0.5, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(3)->rGetLocation()[1], 0.0, 1e-3);

        // Test the elements own the correct nodes
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(0), 0u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(1), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(2), 1u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(3), 2u);

        // Test the element's area and perimeter are computed correctly
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.5, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2 + sqrt(2.0), 1e-6);
    }

    void TestPerformNodeMergeWhenLowIndexNodeMustBeAddedToElement()
    {
        /**
         * Create a mesh comprising two square elements, as shown below. We will test that the
         * nodes marked with an x are merged correctly. We will test node merging in the case
         * where, when the elements previously containing the high-index node are updated to
         * contain the low-index node, at least one of these elements did not already contain
         * the low-index node.
         *
         *   -----x-x---
         *  |     |     |
         *  |     |     |
         *   ----- -----
         *
         * \todo I think this should be a T1 swap (see #1263)
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.00, 0.00));
        nodes.push_back(new Node<2>(1, true, 1.00, 0.00));
        nodes.push_back(new Node<2>(2, true, 2.00, 0.00));
        nodes.push_back(new Node<2>(3, true, 2.00, 1.00));
        nodes.push_back(new Node<2>(4, true, 1.01, 1.00));
        nodes.push_back(new Node<2>(5, true, 1.00, 1.00));
        nodes.push_back(new Node<2>(6, true, 0.00, 2.00));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1;
        unsigned node_indices_elem_0[4] = { 0, 1, 5, 6 };
        unsigned node_indices_elem_1[5] = { 1, 2, 3, 4, 5 };
        for (unsigned i = 0; i < 5; i++)
        {
            if (i < 4)
            {
                nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            }
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Merge nodes 4 and 5
        vertex_mesh.IdentifySwapType(vertex_mesh.GetNode(4), vertex_mesh.GetNode(5));

        // Test the mesh is correctly updated
        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 6u);

        // Test the correct nodes are boundary nodes
        for (unsigned i = 0; i < vertex_mesh.GetNumNodes(); i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetNode(i)->IsBoundaryNode(), true);
        }

        // Test that the moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 1.005, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 1.0, 1e-8);

        // Test the elements own the correct nodes
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 4u);
        unsigned node_indices_element_0[4] = { 0, 1, 4, 5 };
        unsigned node_indices_element_1[4] = { 1, 2, 3, 4 };
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
        }
    }

    void TestPerformT1SwapAndIdentifySwapType()
    {
        /*
         * Create a mesh comprising six nodes contained in two triangle and two rhomboid elements, as shown below.
         * We will test that a T1 swap of the two central nodes is correctly implemented.
         *  _____
         * |\   /|
         * | \ / |
         * |  |  |
         * | / \ |
         * |/___\|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, false, 0.5, 0.4));
        nodes.push_back(new Node<2>(5, false, 0.5, 0.6));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3;
        unsigned node_indices_elem_0[3] = { 2, 3, 5 };
        unsigned node_indices_elem_1[4] = { 2, 5, 4, 1 };
        unsigned node_indices_elem_2[3] = { 1, 4, 0 };
        unsigned node_indices_elem_3[4] = { 0, 4, 5, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            if (i < 3)
            {
                nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
                nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            }
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // For testing edge operations
        vertex_mesh.SetMeshOperationTracking(true);

        // Set the threshold distance between vertices for a T1 swap as follows, to ease calculations
        vertex_mesh.SetCellRearrangementThreshold(0.1 * 2.0 / 1.5);

        // Perform a T1 swap on nodes 4 and 5
        vertex_mesh.IdentifySwapType(vertex_mesh.GetNode(4), vertex_mesh.GetNode(5));

        // Test if the swap has been recorded properly
        auto operation_recorder = vertex_mesh.GetOperationRecorder();
        const std::vector<EdgeOperation>& edge_operations = operation_recorder->GetEdgeOperations();
        const unsigned num_operations = edge_operations.size();

        // Two node merging operations in two elements and two new edge operations in the other two elements
        TS_ASSERT_EQUALS(num_operations, 4u);
        unsigned num_node_merges = 0;
        unsigned num_new_edges = 0;
        for (unsigned i=0; i<num_operations; ++i)
        {
            if (edge_operations[i].GetOperation() == EDGE_OPERATION_NODE_MERGE)
            {
                num_node_merges++;
            }
            if (edge_operations[i].GetOperation() == EDGE_OPERATION_ADD)
            {
                num_new_edges++;
            }
        }
        TS_ASSERT_EQUALS(num_node_merges, 2u);
        TS_ASSERT_EQUALS(num_node_merges, 2u);

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.6, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.5, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.4, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.5, 1e-3);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 2, 3, 5, 4 };
        unsigned node_indices_element_1[3] = { 2, 4, 1 };
        unsigned node_indices_element_2[4] = { 1, 4, 5, 0 };
        unsigned node_indices_element_3[3] = { 0, 5, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNodeGlobalIndex(i), node_indices_element_3[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.3, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.2, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.3, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(3), 0.2, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 1.2 + 0.2 * sqrt(41.0), 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 1.0 + 0.2 * sqrt(41.0), 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 1.2 + 0.2 * sqrt(41.0), 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(3), 1.0 + 0.2 * sqrt(41.0), 1e-6);

        // Test T1 swap information tracking
        std::vector<T1SwapInfo<2> > swap_info
                    = vertex_mesh.GetOperationRecorder()->GetT1SwapsInfo();
        std::vector< c_vector<double, 2> > t1_locations = vertex_mesh.GetLocationsOfT1Swaps();
        //Check location
        // Test T1 swap location tracking
        TS_ASSERT_EQUALS(t1_locations.size(), 1u);
        TS_ASSERT_DELTA(t1_locations[0][0], 0.5, 1e-6);
        TS_ASSERT_DELTA(t1_locations[0][1], 0.5, 1e-6);
        //Check edge orientation
        TS_ASSERT_DELTA(swap_info[0].mPreSwapEdge(0), 0.0, 1e-6);
        TS_ASSERT_DELTA(swap_info[0].mPreSwapEdge(1), 0.2, 1e-6);
        TS_ASSERT_DELTA(swap_info[0].mPostSwapEdge(0), -0.2, 1e-6);
        TS_ASSERT_DELTA(swap_info[0].mPostSwapEdge(1), 0.0, 1e-6);

        OutputFileHandler handler("TestT1InfoArchiving", false);
        std::string archive_filename = handler.GetOutputDirectoryFullPath() + "T1Info.arch";

        {
            // Create an output archive
            std::ofstream ofs(archive_filename.c_str());
            boost::archive::text_oarchive output_arch(ofs);

            // Write VertexMeshOperationRecorder
            output_arch << *vertex_mesh.GetOperationRecorder();
        }

        // Test T1 swap location clearing
        vertex_mesh.ClearLocationsOfT1Swaps();
        swap_info = vertex_mesh.GetOperationRecorder()->GetT1SwapsInfo();
        TS_ASSERT_EQUALS(swap_info.size(), 0u);

        // Retrieve the archive
        {
            // Load T1 swap info
            std::ifstream ifs(archive_filename.c_str(), std::ios::binary);
            boost::archive::text_iarchive input_arch(ifs);

            VertexMeshOperationRecorder<2,2> recorder;
            input_arch >> recorder;
            std::vector<T1SwapInfo<2> > all_swaps = recorder.GetT1SwapsInfo();
            TS_ASSERT_EQUALS(all_swaps.size(), 1u);
            T1SwapInfo<2> info = all_swaps[0];
            TS_ASSERT_DELTA(info.mLocation[0], 0.5, 1e-6);
            TS_ASSERT_DELTA(info.mLocation[1], 0.5, 1e-6);

            TS_ASSERT_DELTA(info.mPreSwapEdge(0), 0.0, 1e-6);
            TS_ASSERT_DELTA(info.mPreSwapEdge(1), 0.2, 1e-6);
            TS_ASSERT_DELTA(info.mPostSwapEdge(0), -0.2, 1e-6);
            TS_ASSERT_DELTA(info.mPostSwapEdge(1), 0.0, 1e-6);
        }
    }

    void TestPerformT1SwapOnBoundary()
    {
        /*
         * Create a mesh comprising six nodes contained in three elements such that all nodes are
         * boundary nodes, as shown below. We will test that a T1 swap is correctly implemented.
         *  _____
         * |\   /
         * | \ /
         * |  |
         * | / \
         * |/___\
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.5, 0.4));
        nodes.push_back(new Node<2>(5, true, 0.5, 0.6));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[3] = { 2, 3, 5 };
        unsigned node_indices_elem_1[3] = { 1, 4, 0 };
        unsigned node_indices_elem_2[4] = { 0, 4, 5, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            if (i < 3)
            {
                nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
                nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            }
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Set the threshold distance between vertices for a T1 swap as follows, to ease calculations
        vertex_mesh.SetCellRearrangementThreshold(0.1 * 2.0 / 1.5);

        // Perform a T1 swap on nodes 5 and 4 (this way round to ensure coverage of boundary node tracking)
        vertex_mesh.IdentifySwapType(vertex_mesh.GetNode(5), vertex_mesh.GetNode(4));

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.6, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.5, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.4, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.5, 1e-3);

        // Test that each element contains the correct number nodes following the rearrangement
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 3u);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 2, 3, 5, 4 };
        unsigned node_indices_element_1[4] = { 1, 4, 5, 0 };
        unsigned node_indices_element_2[4] = { 0, 5, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.3, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.3, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.2, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 1.2 + 0.2 * sqrt(41.0), 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 1.2 + 0.2 * sqrt(41.0), 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 1.0 + 0.2 * sqrt(41.0), 1e-6);

        // Test that the correct nodes are labelled as boundary nodes following the rearrangement
        for (unsigned i = 0; i < vertex_mesh.GetNumNodes(); i++)
        {
            bool expected_boundary_node = (i != 5);
            TS_ASSERT_EQUALS(vertex_mesh.GetNode(i)->IsBoundaryNode(), expected_boundary_node);
        }
    }

    void TestPerformT1SwapOnBoundary2()
    {
        /*
         * Create a mesh comprising six nodes contained in three elements such that all but one node
         * are boundary nodes, as shown below. We will test that a T1 swap is correctly implemented.
         *
         * |\   /|
         * | \ / |
         * |  |  |
         * | / \ |
         * |/___\|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, false, 0.5, 0.4));
        nodes.push_back(new Node<2>(5, true, 0.5, 0.6));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[4] = { 1, 2, 5, 4 };
        unsigned node_indices_elem_1[3] = { 1, 4, 0 };
        unsigned node_indices_elem_2[4] = { 0, 4, 5, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            if (i < 3)
            {
                nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            }
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Set the threshold distance between vertices for a T1 swap as follows, to ease calculations
        vertex_mesh.SetCellRearrangementThreshold(0.1 * 2.0 / 1.5);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 6u);

        // Perform a T1 swap on nodes 5 and 4 (this way round to ensure coverage of boundary node tracking)
        vertex_mesh.IdentifySwapType(vertex_mesh.GetNode(5), vertex_mesh.GetNode(4));

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.6, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.5, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.4, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.5, 1e-3);

        // Test that each element contains the correct number nodes following the rearrangement
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 3u);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[3] = { 1, 2, 4 };
        unsigned node_indices_element_1[4] = { 1, 4, 5, 0 };
        unsigned node_indices_element_2[3] = { 0, 5, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.2, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.3, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.2, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 1.0 + 0.2 * sqrt(41.0), 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 1.2 + 0.2 * sqrt(41.0), 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 1.0 + 0.2 * sqrt(41.0), 1e-6);

        // Test that the correct nodes are labelled as boundary nodes following the rearrangement
        for (unsigned i = 0; i < vertex_mesh.GetNumNodes(); i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetNode(i)->IsBoundaryNode(), true);
        }
    }

    void TestPerformT1SwapWhenVoidForms()
    {
        /*
         * Create a mesh containing six nodes containing in two elements. We will test that
         * a T1 swap is correctly performed in the case where a void forms as a result of
         * the rearrangement, as shown below.
         *
         * |\   /|     |\      /|
         * | \ / |     | \    / |
         * |  |  |  => | /    \ |
         * | / \ |     |/      \|
         * |/   \|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.5, 0.4));
        nodes.push_back(new Node<2>(5, true, 0.5, 0.6));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1;
        unsigned node_indices_elem_0[4] = { 0, 4, 5, 3 };
        unsigned node_indices_elem_1[4] = { 4, 1, 2, 5 };
        for (unsigned i = 0; i < 4; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Set the threshold distance between vertices for a T1 swap as follows, to ease calculations
        vertex_mesh.SetCellRearrangementThreshold(0.1 * 2.0 / 1.5);

        // Perform a T1 swap on nodes 5 and 4.
        vertex_mesh.IdentifySwapType(vertex_mesh.GetNode(5), vertex_mesh.GetNode(4));

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.6, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.5, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.4, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.5, 1e-3);

        // Test that each element contains the correct number of nodes following the rearrangement
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[3] = { 0, 5, 3 };
        unsigned node_indices_element_1[3] = { 4, 1, 2 };
        for (unsigned i = 0; i < 3; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.2, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.2, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 1.0 + 0.2 * sqrt(41.0), 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 1.0 + 0.2 * sqrt(41.0), 1e-6);

        // Test that the correct nodes are labelled as boundary nodes following the rearrangement
        for (unsigned i = 0; i < vertex_mesh.GetNumNodes(); i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetNode(i)->IsBoundaryNode(), true);
        }
    }

    void TestPerformT1SwapExceptions()
    {
        /*
         * Create a mesh comprising six nodes containing in two triangle and two rhomboid elements,
         * where two nodes (those with indices 4 and 5) have the same location. We will test that
         * trying to perform a T1 swap on these nodes throws the correct exception.
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, false, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, false, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, false, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, false, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, false, 0.5, 0.5));
        nodes.push_back(new Node<2>(5, false, 0.5, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3;
        unsigned node_indices_elem_0[3] = { 2, 3, 5 };
        unsigned node_indices_elem_1[4] = { 2, 5, 4, 1 };
        unsigned node_indices_elem_2[3] = { 1, 4, 0 };
        unsigned node_indices_elem_3[4] = { 0, 4, 5, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            if (i < 3)
            {
                nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
                nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            }
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Set the threshold distance between vertices for a T1 swap as follows, to ease calculations
        vertex_mesh.SetCellRearrangementThreshold(0.1 * 2.0 / 1.5);

        // Test that trying to perform a T1 swap on nodes 4 and 5 throws the correct exception
        TS_ASSERT_THROWS_THIS(vertex_mesh.IdentifySwapType(vertex_mesh.GetNode(4), vertex_mesh.GetNode(5)), "Nodes are too close together, this shouldn't happen");
    }

    void TestPerformT1SwapWithAddingEdgeToTriangularElement()
    {
        /**
         * Create a mesh comprising six nodes contained in two triangle and two rhomboid elements, as shown below.
         * We will test that the triangles are allowed to gain an edge.
         *  _____
         * |\   /|
         * | \ / |
         * |  |  |
         * | / \ |
         * |/___\|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, false, 0.5, 0.4));
        nodes.push_back(new Node<2>(5, false, 0.5, 0.6));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3;
        unsigned node_indices_elem_0[3] = { 2, 3, 5 };
        unsigned node_indices_elem_1[4] = { 2, 5, 4, 1 };
        unsigned node_indices_elem_2[3] = { 1, 4, 0 };
        unsigned node_indices_elem_3[4] = { 0, 4, 5, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            if (i < 3)
            {
                nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
                nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            }
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Ensure that the inner node will swap
        vertex_mesh.SetCellRearrangementThreshold(0.21);

        // Perform a T1 swap on nodes 4 and 5
        vertex_mesh.CheckForSwapsFromShortEdges();

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 2, 3, 5, 4 };
        unsigned node_indices_element_1[3] = { 2, 4, 1 };
        unsigned node_indices_element_2[4] = { 1, 4, 5, 0 };
        unsigned node_indices_element_3[3] = { 0, 5, 3 };

        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNodeGlobalIndex(i), node_indices_element_3[i]);
            }
        }
    }

    void TestRemoveTriangularVoidWithRemovingEdgeFromInternalTriangularVoid()
    {
        /**
         * In this test we check that a T1 swap occurs if we have a triangular void
         *
         *       ____________
         *      |            |\
         *      |____________| \  Where the small triangles are voids so the bottom
         *      |  |/   \|   | /  Element is (slightly) convex
         *      |____________|/
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 2.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 2.0, 2.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 2.0));
        nodes.push_back(new Node<2>(4, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(5, true, 0.5, 1.0));
        nodes.push_back(new Node<2>(6, true, 0.7, 1.0));
        nodes.push_back(new Node<2>(7, true, 1.3, 1.0));
        nodes.push_back(new Node<2>(8, true, 1.5, 1.0));
        nodes.push_back(new Node<2>(9, false, 2.0, 1.0));
        nodes.push_back(new Node<2>(10, true, 2.5, 1.0));
        nodes.push_back(new Node<2>(11, true, 0.5, 0.9));
        nodes.push_back(new Node<2>(12, true, 1.5, 0.9));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[10] = {0, 1, 9, 8, 12, 7, 6, 11, 5, 4};
        unsigned node_indices_elem_1[8] = {4, 5, 6, 7, 8, 9, 2, 3};
        unsigned node_indices_elem_2[4] = {1, 10, 2, 9};


        for (unsigned i=0; i<10; i++)
        {
            if (i < 8)
            {
                nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            }
            if (i < 4)
            {
                nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            }
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Ensure that the inner edge will be considered for a swap
        vertex_mesh.SetCellRearrangementThreshold(0.11);

        // Check for T1 swaps and carry them out if allowed - the short edge on the right should swap and remove the void
        vertex_mesh.CheckForSwapsFromShortEdges();

        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 7u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 6u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 4u);

        // Check for T1 swaps and carry them out if allowed - the short edge on the left should swap and remove the void
        vertex_mesh.CheckForSwapsFromShortEdges();

        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 4u);

        // Test boundary property of nodes. All are boundary nodes except node 3.
        for (unsigned i=0; i<vertex_mesh.GetNumNodes(); i++)
        {
            bool expected_boundary_node = (i!=5);
            TS_ASSERT_EQUALS(vertex_mesh.GetNode(i)->IsBoundaryNode(), expected_boundary_node);
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 2.0, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 2, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.5, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 6.0, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 6.0, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 2.0+sqrt(5.0), 1e-6);
    }

    void TestDoNotPerformT1SwapWithRemovingEdgeFromTriangularElement()
    {
        /**
         * In this test we check that a T1 swap does not occur if one of the elements is triangular
         * and would loose an edge by swapping nodes. The mesh looks like this
         *
         *       ______________
         *      |\             |
         *      | \ _________  |
         *      |  |          \| ...where the funny shaped element in the middle is supposed to be
         *      |  |_________ /|    a very long triangle that has the third vertex on the right hand boundary.
         *      | /            |
         *      |/_____________|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 2.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 2.0, 2.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 2.0));
        nodes.push_back(new Node<2>(4, false, 0.2, 0.95));
        nodes.push_back(new Node<2>(5, true, 2.0, 1.0));
        nodes.push_back(new Node<2>(6, false, 0.2, 1.05));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3;
        unsigned node_indices_elem_0[4] = { 0, 1, 5, 4 };
        unsigned node_indices_elem_1[4] = { 5, 2, 3, 6 };
        unsigned node_indices_elem_2[4] = { 0, 4, 6, 3 };
        unsigned node_indices_elem_3[3] = { 4, 5, 6 };
        for (unsigned i = 0; i < 4; i++)
        {
            if (i < 3)
            {
                nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
            }
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Ensure that the inner edge will be considered for a swap
        vertex_mesh.SetCellRearrangementThreshold(0.11);

        // Check for T1 swaps and carry them out if allowed - the short edge should not swap!
        vertex_mesh.CheckForSwapsFromShortEdges();

        TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNumNodes(), 3u);

        // Test that each element still contains the correct nodes following the rearrangement
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_elem_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_elem_1[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_elem_2[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNodeGlobalIndex(i), node_indices_elem_3[i]);
            }
        }
    }

    void TestExceptionForVoidRemovalWithRemovingEdgeFromTriangularElement()
    {
        /**
         * In this test we check that void removal does not occur if one of the adjacent elements is triangular
         * and would loose an edge by swapping nodes. The code should throw and exception in this case.
         * The mesh looks like this
         *
         *       ______________./This corner is not a node.
         *      |\      1      |
         *      | \ _________  |
         *      |  |   void   \| ...where elements 1, and 2 are triangles that share the right hand vertex
         *      |  |_________ /|    with the triangular void in the middle.
         *      | /     2      |
         *      |/_____________|.This corner is not a node either.
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 0.0, 2.0));
        nodes.push_back(new Node<2>(2, true, 0.2, 0.95));
        nodes.push_back(new Node<2>(3, true, 2.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.2, 1.05));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[4] = { 0, 2, 4, 1 };
        unsigned node_indices_elem_1[3] = { 1, 4, 3 };
        unsigned node_indices_elem_2[3] = { 0, 3, 2 };

        for (unsigned i = 0; i < 4; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            if (i < 3)
            {
                nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
                nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            }
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2,2> vertex_mesh(nodes, vertex_elements);
        vertex_mesh.SetMeshOperationTracking(true);//For coverage

        // Ensure that the inner edge will be considered for a swap
        vertex_mesh.SetCellRearrangementThreshold(0.11);

        // Check for possible swaps and carry them out if allowed - the short edge should not swap and
        // the void should not be removed!
        TS_ASSERT_THROWS_THIS(vertex_mesh.CheckForSwapsFromShortEdges(),
                              "Triangular element next to triangular void, not implemented yet.");
    }

    void TestPerformT2Swap()
    {
        /*
         * Create a mesh comprising six nodes contained in three trapezium element and
         * a central triangle element, as shown below. We will test that a T2 swap
         * correctly removes the triangle element from the mesh.
         *
         *      /|\
         *     / | \
         *    /  |  \    (the triangular element has index zero)
         *   /2 /_\ 1\
         *  /  /   \  \
         * /__/__3__\__\
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 0.5, 0.5));
        nodes.push_back(new Node<2>(3, false, 0.4, 0.25));
        nodes.push_back(new Node<2>(4, false, 0.6, 0.25));
        nodes.push_back(new Node<2>(5, false, 0.5, 0.3));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3;
        unsigned node_indices_elem_0[3] = {3, 4, 5};
        unsigned node_indices_elem_1[4] = {1, 2, 5, 4};
        unsigned node_indices_elem_2[4] = {2, 0, 3, 5};
        unsigned node_indices_elem_3[4] = {0, 1, 4, 3};
        for (unsigned i=0; i<4; i++)
        {
            if (i < 3)
            {
                nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            }
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
        }

        std::vector<VertexElement<2,2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2,2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2,2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2,2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2,2>(3, nodes_elem_3));

        MutableVertexMesh<2,2> vertex_mesh(nodes, vertex_elements);
        vertex_mesh.SetMeshOperationTracking(true);
        vertex_mesh.SetMeshOperationTracking(true);

        // Perform a T2 swap on the central triangle element
        VertexElement<2,2>* p_element_0 = vertex_mesh.GetElement(0);
        c_vector<double, 2> centroid_of_element_0_before_swap = vertex_mesh.GetCentroidOfElement(0);
        vertex_mesh.PerformT2Swap(*p_element_0);

        // Test if the swap has been recorded properly
        auto operation_recorder = vertex_mesh.GetOperationRecorder();
        const std::vector<EdgeOperation>& edge_operations = operation_recorder->GetEdgeOperations();
        const unsigned num_operations = edge_operations.size();

        // Two node merging operations in two elements and two new edge operations in the other two elements
        TS_ASSERT_EQUALS(num_operations, 3u);
        unsigned num_node_merges = 0;
        for (unsigned i=0; i<num_operations; ++i)
        {
            if (edge_operations[i].GetOperation() == EDGE_OPERATION_NODE_MERGE)
            {
                num_node_merges++;
            }
        }
        TS_ASSERT_EQUALS(num_node_merges, 3u);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 4u);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumAllElements(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumAllNodes(), 7u);

        for (unsigned j=1; j<4; j++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(j)->GetNumNodes(), 3u);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(j)->GetNodeGlobalIndex(0), j%3);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(j)->GetNodeGlobalIndex(1), (j+1)%3);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(j)->GetNodeGlobalIndex(2), 6u);
        }

        // Test boundary property of nodes. All are boundary nodes except node 3.
        for (unsigned i=0; i<vertex_mesh.GetNumNodes(); i++)
        {
            bool expected_boundary_node = (i!=3);
            TS_ASSERT_EQUALS(vertex_mesh.GetNode(i)->IsBoundaryNode(), expected_boundary_node);
        }

        // Test the location of the new node:
        TS_ASSERT_DELTA(vertex_mesh.GetNode(6)->rGetLocation()[0], centroid_of_element_0_before_swap[0], 1e-10);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(6)->rGetLocation()[1], centroid_of_element_0_before_swap[1], 1e-10);

        // Test the tracking of the T2 swap location:
        TS_ASSERT_DELTA(vertex_mesh.GetLastT2SwapLocation()[0], centroid_of_element_0_before_swap[0], 1e-10);
        TS_ASSERT_DELTA(vertex_mesh.GetLastT2SwapLocation()[1], centroid_of_element_0_before_swap[1], 1e-10);

        // Test T2 swap information tracking
        std::vector<T2SwapInfo<2> > swap_info
        = vertex_mesh.GetOperationRecorder()->GetT2SwapsInfo();

        //Check location
        TS_ASSERT_EQUALS(swap_info.size(), 1u);
        TS_ASSERT_DELTA(swap_info[0].mLocation(0), centroid_of_element_0_before_swap[0], 1e-10);
        TS_ASSERT_DELTA(swap_info[0].mLocation(1), centroid_of_element_0_before_swap[1], 1e-10);
        //Check cell ID of T2 swapped cell
        TS_ASSERT_EQUALS(swap_info[0].mCellId, 0);

        OutputFileHandler handler("TestT2InfoArchiving", false);
        std::string archive_filename = handler.GetOutputDirectoryFullPath() + "T2Info.arch";

        {
            // Create an output archive
            std::ofstream ofs(archive_filename.c_str());
            boost::archive::text_oarchive output_arch(ofs);

            // Write the OperationRecorder object with the swap info
            output_arch << *vertex_mesh.GetOperationRecorder();
        }

        // Test T2 swap info clearing
        vertex_mesh.GetOperationRecorder()->ClearT2SwapsInfo();
        swap_info = vertex_mesh.GetOperationRecorder()->GetT2SwapsInfo();
        TS_ASSERT_EQUALS(swap_info.size(), 0u);

        // Retrieve the archive
        {
            // Load T2 swap info
            std::ifstream ifs(archive_filename.c_str(), std::ios::binary);
            boost::archive::text_iarchive input_arch(ifs);

            VertexMeshOperationRecorder<2,2> recorder;
            input_arch >> recorder;
            std::vector<T2SwapInfo<2> > all_swaps = recorder.GetT2SwapsInfo();
            TS_ASSERT_EQUALS(all_swaps.size(), 1u);
            T2SwapInfo<2> info = all_swaps[0];

            TS_ASSERT_DELTA(info.mLocation[0], centroid_of_element_0_before_swap[0], 1e-10);
            TS_ASSERT_DELTA(info.mLocation[1], centroid_of_element_0_before_swap[1], 1e-10);

            TS_ASSERT_EQUALS(info.mCellId, 0);
        }
    }

    void TestPerformT2SwapWithBoundaryNodes()
    {
        /*
         * Create a mesh comprising six nodes contained in two trapezium elements
         * and one triangle element, as shown below. We will test that a T2 swap
         * is performed correctly when boundary nodes are involved.
         *
         *       /|\
         *      / | \
         *     /  |  \    (the triangular element has index zero)
         *    /2 /_\ 1\
         *   /  /   \  \
         *  /__/     \__\
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 0.5, 0.5));
        nodes.push_back(new Node<2>(3, true, 0.4, 0.25));
        nodes.push_back(new Node<2>(4, true, 0.6, 0.25));
        nodes.push_back(new Node<2>(5, false, 0.5, 0.3));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[3] = { 3, 4, 5 };
        unsigned node_indices_elem_1[4] = { 1, 2, 5, 4 };
        unsigned node_indices_elem_2[4] = { 2, 0, 3, 5 };
        for (unsigned i = 0; i < 4; i++)
        {
            if (i < 3)
            {
                nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            }
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 6u);

        // Perform a T2 swap on the central triangle element
        VertexElement<2, 2>* p_element_0 = vertex_mesh.GetElement(0);
        vertex_mesh.PerformT2Swap(*p_element_0);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 4u);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumAllElements(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumAllNodes(), 7u);

        for (unsigned j = 1; j < 3; j++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(j)->GetNumNodes(), 3u);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(j)->GetNodeGlobalIndex(0), j % 3);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(j)->GetNodeGlobalIndex(1), (j + 1) % 3);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(j)->GetNodeGlobalIndex(2), 6u);
        }

        // Test boundary property of nodes. All are boundary nodes.
        for (unsigned i = 0; i < vertex_mesh.GetNumNodes(); i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetNode(i)->IsBoundaryNode(), true);
        }

        // Make five nodes to assign to two elements
        std::vector<Node<2>*> nodes2;
        nodes2.push_back(new Node<2>(0, true, 1.0, 0.0));
        nodes2.push_back(new Node<2>(1, true, 0.5, 0.5));
        nodes2.push_back(new Node<2>(2, true, 0.4, 0.25));
        nodes2.push_back(new Node<2>(3, true, 0.6, 0.25));
        nodes2.push_back(new Node<2>(4, true, 0.5, 0.3));

        /*
         *  Make one trapezium element with a central triangular element out of these nodes
         *
         *       |\
         *       | \
         *       |  \
         *      /_\ 1\   Triangular element is element zero
         *         \_ \
         *           \_\
         *
         */

        // Triangle element
        std::vector<Node<2>*> nodes2_elem_0;
        nodes2_elem_0.push_back(nodes2[2]);
        nodes2_elem_0.push_back(nodes2[3]);
        nodes2_elem_0.push_back(nodes2[4]);

        // Trapezium
        std::vector<Node<2>*> nodes2_elem_1;
        nodes2_elem_1.push_back(nodes2[0]);
        nodes2_elem_1.push_back(nodes2[1]);
        nodes2_elem_1.push_back(nodes2[4]);
        nodes2_elem_1.push_back(nodes2[3]);

        std::vector<VertexElement<2, 2>*> vertex_elements2;
        vertex_elements2.push_back(new VertexElement<2, 2>(0, nodes2_elem_0));
        vertex_elements2.push_back(new VertexElement<2, 2>(1, nodes2_elem_1));

        // Make a vertex mesh
        MutableVertexMesh<2, 2> vertex_mesh2(nodes2, vertex_elements2);

        TS_ASSERT_EQUALS(vertex_mesh2.GetNumElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetNumNodes(), 5u);

        // Perform a T2 swap on the middle triangle element
        p_element_0 = vertex_mesh2.GetElement(0);
        vertex_mesh2.PerformT2Swap(*p_element_0);

        TS_ASSERT_EQUALS(vertex_mesh2.GetNumElements(), 1u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetNumNodes(), 3u);

        TS_ASSERT_EQUALS(vertex_mesh2.GetNumAllElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetNumAllNodes(), 6u);

        TS_ASSERT_EQUALS(vertex_mesh2.GetElement(1)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetElement(1)->GetNodeGlobalIndex(0), 0u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetElement(1)->GetNodeGlobalIndex(1), 1u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetElement(1)->GetNodeGlobalIndex(2), 5u);

        // Test boundary property of nodes. All are boundary nodes.
        for (unsigned i = 0; i < vertex_mesh2.GetNumNodes(); i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh2.GetNode(i)->IsBoundaryNode(), true);
        }
    }

    void TestT2SwapsDontOccurWithTriangularNeighbours()
    {
        // Make 6 nodes to assign to four elements
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, false, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, false, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, false, 0.5, 0.5));
        nodes.push_back(new Node<2>(3, false, 0.4, 0.25));
        nodes.push_back(new Node<2>(4, false, 0.6, 0.25));
        nodes.push_back(new Node<2>(5, false, 0.5, 0.3));

        // Make two triangles and two trapezium elements out of these nodes
        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3;
        unsigned node_indices_elem_0[3] = { 3, 4, 5 };
        unsigned node_indices_elem_1[3] = { 2, 5, 4 };
        unsigned node_indices_elem_2[4] = { 2, 0, 3, 5 };
        unsigned node_indices_elem_3[4] = { 0, 1, 4, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            if (i < 3)
            {
                nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
                nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            }
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements, 0.1);

        // Set the threshold distance between vertices for a T1 swap as follows, to ease calculations
        vertex_mesh.SetCellRearrangementThreshold(0.1);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 4u);

        // Attempt to perform a T2 swap on the middle triangle element
        VertexElement<2, 2>* p_element_0 = vertex_mesh.GetElement(0);
        TS_ASSERT_THROWS_THIS(vertex_mesh.PerformT2Swap(*p_element_0),
                              "One of the neighbours of a small triangular element is also a triangle - "
                              "dealing with this has not been implemented yet");
    }

    void TestPerformT2SwapWithRosettes()
    {
        /* Create a mesh containing a smaller triangular element, each of whose nodes are
         * 'rosette' nodes. Test that a T2 swap correctly removes the triangular element
         * from the mesh.
         *  _________
         *  |\      |
         *  | \     |
         *  | |_\___|
         *  | / |   |
         *  |/__|___|
         */

        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 2.0, 0.0));
        nodes.push_back(new Node<2>(3, false, 0.99, 1.0));
        nodes.push_back(new Node<2>(4, false, 1.0, 1.0));
        nodes.push_back(new Node<2>(5, false, 2.0, 1.0));
        nodes.push_back(new Node<2>(6, false, 0.99, 1.01));
        nodes.push_back(new Node<2>(7, false, 0.0, 2.0));
        nodes.push_back(new Node<2>(8, false, 2.0, 2.0));

        std::vector<Node<2>*> nodes_elem_0;
        nodes_elem_0.push_back(nodes[0]);
        nodes_elem_0.push_back(nodes[1]);
        nodes_elem_0.push_back(nodes[4]);
        nodes_elem_0.push_back(nodes[3]);

        std::vector<Node<2>*> nodes_elem_1;
        nodes_elem_1.push_back(nodes[1]);
        nodes_elem_1.push_back(nodes[2]);
        nodes_elem_1.push_back(nodes[5]);
        nodes_elem_1.push_back(nodes[4]);

        std::vector<Node<2>*> nodes_elem_2;
        nodes_elem_2.push_back(nodes[0]);
        nodes_elem_2.push_back(nodes[3]);
        nodes_elem_2.push_back(nodes[6]);
        nodes_elem_2.push_back(nodes[7]);

        std::vector<Node<2>*> nodes_elem_3;
        nodes_elem_3.push_back(nodes[3]);
        nodes_elem_3.push_back(nodes[4]);
        nodes_elem_3.push_back(nodes[6]);

        std::vector<Node<2>*> nodes_elem_4;
        nodes_elem_4.push_back(nodes[4]);
        nodes_elem_4.push_back(nodes[5]);
        nodes_elem_4.push_back(nodes[8]);
        nodes_elem_4.push_back(nodes[7]);
        nodes_elem_4.push_back(nodes[6]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));
        vertex_elements.push_back(new VertexElement<2, 2>(4, nodes_elem_4));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Perform a T2 swap on the central triangle element
        VertexElement<2, 2>* p_element_3 = vertex_mesh.GetElement(3);
        c_vector<double, 2> centroid_of_element_0_before_swap = vertex_mesh.GetCentroidOfElement(3);
        vertex_mesh.PerformT2Swap(*p_element_3);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 7u);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumAllElements(), 5u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumAllNodes(), 10u);

        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(0), 0u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(1), 1u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(2), 9u);

        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(0), 1u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(1), 2u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(2), 5u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(3), 9u);

        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(0), 0u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(1), 9u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(2), 7u);

        TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNodeGlobalIndex(0), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNodeGlobalIndex(1), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNodeGlobalIndex(2), 6u);

        TS_ASSERT_EQUALS(vertex_mesh.GetElement(4)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(4)->GetNodeGlobalIndex(0), 9u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(4)->GetNodeGlobalIndex(1), 5u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(4)->GetNodeGlobalIndex(2), 8u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(4)->GetNodeGlobalIndex(3), 7u);
    }

    void TestPerformT2SwapWithoutNeighbours()
    {
        /*
         * Create a mesh comprising three nodes contained in one triangle
         * element, as shown below. We will test that a T2 swap is performed
         * correctly when an element has no neighbouring elements.
         *        _
         *       / \
         *      /   \
         *     /     \
         *    /   0   \
         *   /         \
         *  /___________\
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 0.2, 0.0));
        nodes.push_back(new Node<2>(2, true, 0.1, 0.05));

        std::vector<Node<2>*> nodes_elem_0;
        nodes_elem_0.push_back(nodes[0]);
        nodes_elem_0.push_back(nodes[1]);
        nodes_elem_0.push_back(nodes[2]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 1u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 3u);

        // Perform a T2 swap on the central triangle element
        auto p_element_0 = vertex_mesh.GetElement(0);
        vertex_mesh.PerformT2Swap(*p_element_0);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 0u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 0u);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumAllElements(), 1u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumAllNodes(), 3u);

        /*
         *  Make two disconnected triangular elements
         *        _
         *       / \                 / \
         *      /   \               /   \
         *     /     \             /     \
         *    /   0   \           /   1   \
         *   /         \         /         \
         *  /___________\       /___________\
         *
         */
        std::vector<Node<2>*> nodes2;
        nodes2.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes2.push_back(new Node<2>(1, true, 0.2, 0.0));
        nodes2.push_back(new Node<2>(2, true, 0.1, 0.05));
        nodes2.push_back(new Node<2>(3, true, 1.0, 0.0));
        nodes2.push_back(new Node<2>(4, true, 1.2, 0.0));
        nodes2.push_back(new Node<2>(5, true, 1.1, 0.05));

        std::vector<Node<2>*> nodes2_elem_0;
        nodes2_elem_0.push_back(nodes2[0]);
        nodes2_elem_0.push_back(nodes2[1]);
        nodes2_elem_0.push_back(nodes2[2]);

        std::vector<Node<2>*> nodes2_elem_1;
        nodes2_elem_1.push_back(nodes2[3]);
        nodes2_elem_1.push_back(nodes2[4]);
        nodes2_elem_1.push_back(nodes2[5]);

        std::vector<VertexElement<2, 2>*> vertex_elements2;
        vertex_elements2.push_back(new VertexElement<2, 2>(0, nodes2_elem_0));
        vertex_elements2.push_back(new VertexElement<2, 2>(1, nodes2_elem_1));

        // Make a vertex mesh
        MutableVertexMesh<2, 2> vertex_mesh2(nodes2, vertex_elements2);

        TS_ASSERT_EQUALS(vertex_mesh2.GetNumElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetNumNodes(), 6u);

        // Perform a T2 swap on the left triangle element
        p_element_0 = vertex_mesh2.GetElement(0);
        vertex_mesh2.PerformT2Swap(*p_element_0);

        TS_ASSERT_EQUALS(vertex_mesh2.GetNumElements(), 1u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetNumNodes(), 3u);

        TS_ASSERT_EQUALS(vertex_mesh2.GetNumAllElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetNumAllNodes(), 6u);

        TS_ASSERT_EQUALS(vertex_mesh2.GetElement(1)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetElement(1)->GetNodeGlobalIndex(0), 3u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetElement(1)->GetNodeGlobalIndex(1), 4u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetElement(1)->GetNodeGlobalIndex(2), 5u);

        // Test boundary property of nodes. All are boundary nodes.
        for (unsigned i = 0; i < vertex_mesh2.GetNumNodes(); i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh2.GetNode(i)->IsBoundaryNode(), true);
        }

        // Perform a T2 swap on the remaining triangle element
        auto p_element_1 = vertex_mesh2.GetElement(1);
        vertex_mesh2.PerformT2Swap(*p_element_1);

        TS_ASSERT_EQUALS(vertex_mesh2.GetNumElements(), 0u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetNumNodes(), 0u);

        TS_ASSERT_EQUALS(vertex_mesh2.GetNumAllElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh2.GetNumAllNodes(), 6u);
    }

    void TestReMeshForT1Swaps()
    {
        /*
         * Read in a vertex mesh that contains several pairs of nodes that are close enough for
         * T1 swaps to be performed, as shown below. The mesh consists of six elements and all
         * T1 swaps are performed on all horizontal edges. We will test that the ReMesh() method
         * correctly performs T1 swaps for internal and boundary elements, and correctly updates
         * which nodes are labelled as boundary nodes.
         *
         *      /\    /\
         *     /  \__/  \
         *    /   /  \   \
         *    \__/\__/\__/
         *    /  \/  \/  \
         *    \   \__/   /
         *     \  /  \  /
         *      \/    \/
         */
        VertexMeshReader<2, 2> mesh_reader("cell_based/test/data/TestMutableVertexMesh/vertex_remesh_T1");
        MutableVertexMesh<2, 2> vertex_mesh;

        vertex_mesh.ConstructFromMeshReader(mesh_reader);
        vertex_mesh.SetCellRearrangementThreshold(0.1);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 8u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 22u);

        // Calls ReMesh() to identify and perform any T1 swaps
        vertex_mesh.ReMesh();

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 8u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 22u);

        std::string dirname = "TestVertexMeshReMesh";
        std::string mesh_filename = "vertex_remesh_T1";

        // Save the mesh data using mesh writers
        VertexMeshWriter<2, 2> mesh_writer(dirname, mesh_filename, false);
        mesh_writer.WriteFilesUsingMesh(vertex_mesh);

        // Check the positions are updated correctly
        OutputFileHandler handler("TestVertexMeshReMesh", false);
        std::string results_file1 = handler.GetOutputDirectoryFullPath() + "vertex_remesh_T1.node";
        std::string results_file2 = handler.GetOutputDirectoryFullPath() + "vertex_remesh_T1.cell";

        FileComparison comparer1(results_file1, "cell_based/test/data/TestMutableVertexMesh/vertex_remesh_T1_after_remesh.node");
        TS_ASSERT(comparer1.CompareFiles());
        FileComparison comparer2(results_file2, "cell_based/test/data/TestMutableVertexMesh/vertex_remesh_T1_after_remesh.cell");
        TS_ASSERT(comparer2.CompareFiles());
    }

    void TestReMeshExceptionWhenNonBoundaryNodesAreContainedOnlyInTwoElements()
    {
        /*
         * Create a mesh comprising six nodes contained in two elements, as shown below. We will test
         * that when we attempt to mesh the nodes marked x, the correct exception is thrown.
         *   ______
         *  |     /|
         *  |    / |
         *  |   x  |
         *  |  x   |
         *  | /    |
         *  |/_____|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.00, 0.00));
        nodes.push_back(new Node<2>(1, true, 1.00, 0.00));
        nodes.push_back(new Node<2>(2, true, 1.00, 1.00));
        nodes.push_back(new Node<2>(3, true, 0.00, 1.00));
        nodes.push_back(new Node<2>(4, false, 0.49, 0.49));
        nodes.push_back(new Node<2>(5, false, 0.51, 0.51));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1;
        unsigned node_indices_elem_0[5] = { 0, 1, 2, 5, 4 };
        unsigned node_indices_elem_1[5] = { 0, 4, 5, 2, 3 };
        for (unsigned i = 0; i < 5; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);
        vertex_mesh.SetCellRearrangementThreshold(0.1);

        TS_ASSERT_THROWS_THIS(vertex_mesh.ReMesh(), "There are non-boundary nodes contained only in two elements; something has gone wrong.");
    }

    void TestIdentifySwapTypeExceptionWhenBoundaryNodeIsContainedInThreeElements()
    {
        /*
         * Create a mesh as shown below, where the two nodes marked with an x are to be merged.
         * We will test that the correct exception is thrown when IdentifySwapType() is called.
         *             _____
         *           /      |
         *          /       |
         *   -----xx--------|
         *  |       \       |
         *  |________\______|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.00, 0.0));
        nodes.push_back(new Node<2>(1, true, 2.00, 0.0));
        nodes.push_back(new Node<2>(2, true, 3.00, 0.0));
        nodes.push_back(new Node<2>(3, true, 3.00, 1.0));
        nodes.push_back(new Node<2>(4, true, 3.00, 2.0));
        nodes.push_back(new Node<2>(5, true, 2.00, 2.0));
        nodes.push_back(new Node<2>(6, true, 1.00, 1.0));
        nodes.push_back(new Node<2>(7, true, 0.99, 1.0));
        nodes.push_back(new Node<2>(8, true, 0.00, 1.0));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[5] = { 0, 1, 6, 7, 8 };
        unsigned node_indices_elem_1[4] = { 1, 2, 3, 6 };
        unsigned node_indices_elem_2[4] = { 3, 4, 5, 6 };
        for (unsigned i = 0; i < 4; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
        }
        nodes_elem_0.push_back(nodes[node_indices_elem_0[4]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);
        vertex_mesh.SetCellRearrangementThreshold(0.1);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 9u);

        TS_ASSERT_THROWS_THIS(vertex_mesh.IdentifySwapType(vertex_mesh.GetNode(6), vertex_mesh.GetNode(7)), "There is a boundary node contained in three elements something has gone wrong.");
    }

    void TestReMeshExceptionWhenNonBoundaryNodeIsContainedOnlyInTwoElements()
    {
        /*
         * Create a mesh as shown below, where the two nodes marked with an x are to be merged.
         * We will test that the correct exception is thrown when ReMesh() is called.
         *
         * |\   /|
         * | \ / |
         * |  x  |
         * |  x  |
         * |__|__|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, false, 0.5, 0.49));
        nodes.push_back(new Node<2>(5, true, 0.5, 0.51));
        nodes.push_back(new Node<2>(6, true, 0.5, 0.0));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1;
        unsigned node_indices_elem_0[5] = { 1, 2, 5, 4, 6 };
        unsigned node_indices_elem_1[5] = { 0, 6, 4, 5, 3 };
        for (unsigned i = 0; i < 5; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);
        vertex_mesh.SetCellRearrangementThreshold(0.1);

        TS_ASSERT_THROWS_THIS(vertex_mesh.ReMesh(), "There is a non-boundary node contained only in two elements; something has gone wrong.");
    }

    void TestAnotherReMeshExceptionWhenNonBoundaryNodesAreContainedOnlyInTwoElements()
    {
        /*
         * Create a mesh as shown below, where the two central nodes marked with an x are to be merged. We will
         * test that the correct exception is thrown when ReMesh() is called. Note that the extra node at the
         * top of the mesh is required to stop the element from containing only three nodes, otherwise the
         * ReMesh() method would not call IdentifySwapType().
         *
         *  __x__
         * |\   /|
         * | \ / |
         * |  x  |
         * |  x  |
         * |__|__|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.00));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.00));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.00));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.00));
        nodes.push_back(new Node<2>(4, false, 0.5, 0.49));
        nodes.push_back(new Node<2>(5, false, 0.5, 0.51));
        nodes.push_back(new Node<2>(6, true, 0.5, 0.00));
        nodes.push_back(new Node<2>(7, true, 0.5, 1.00));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[5] = { 1, 2, 5, 4, 6 };
        unsigned node_indices_elem_1[5] = { 0, 6, 4, 5, 3 };
        unsigned node_indices_elem_2[4] = { 2, 3, 5, 7 };
        for (unsigned i = 0; i < 4; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
        }
        nodes_elem_0.push_back(nodes[node_indices_elem_0[4]]);
        nodes_elem_1.push_back(nodes[node_indices_elem_1[4]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);
        vertex_mesh.SetCellRearrangementThreshold(0.1);

        TS_ASSERT_THROWS_THIS(vertex_mesh.ReMesh(), "There are non-boundary nodes contained only in two elements; something has gone wrong.");
    }

    void TestPerformT3Swap()
    {
        /*
         * Create a mesh comprising 13 nodes containing in five elements, as shown below.
         * We will test that a T3 swap is correctly performed.
         *         _____
         *    |\  |     |  /|
         *    | \ |     | /_|
         *    | / |     | \ |
         *    |/__|_____|__\|
         *    |             |
         *    |_____________|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true,   0.0,  0.0));
        nodes.push_back(new Node<2>(1, true,   1.0,  0.0));
        nodes.push_back(new Node<2>(2, true,   1.0,  1.0));
        nodes.push_back(new Node<2>(3, true,   0.0,  1.0));
        nodes.push_back(new Node<2>(4, true,   2.0,  0.0));
        nodes.push_back(new Node<2>(5, true,   2.0,  1.0));
        nodes.push_back(new Node<2>(6, true,   1.1,  0.5));
        nodes.push_back(new Node<2>(7, true,  -1.0,  0.0));
        nodes.push_back(new Node<2>(8, true,  -0.1,  0.5));
        nodes.push_back(new Node<2>(9, true,  -1.0,  1.0));
        nodes.push_back(new Node<2>(10, true, -1.0, -1.0));
        nodes.push_back(new Node<2>(11, true,  2.0, -1.0));
        nodes.push_back(new Node<2>(12, true,  2.0,  0.5));

        std::vector<Node<2>*> nodes_in_element0, nodes_in_element1, nodes_in_element2, nodes_in_element3, nodes_in_element4;
        unsigned node_indices_element_0[4] = {0, 1, 2, 3};
        unsigned node_indices_element_1[3] = {4, 12, 6};
        unsigned node_indices_element_2[3] = {12, 5, 6};
        unsigned node_indices_element_3[3] = {7, 8, 9};
        unsigned node_indices_element_4[6] = {10, 11, 4, 1, 0, 7};
        for (unsigned i=0; i<6; i++)
        {
            if (i < 4)
            {
                nodes_in_element0.push_back(nodes[node_indices_element_0[i]]);
            }
            if (i < 3)
            {
                nodes_in_element1.push_back(nodes[node_indices_element_1[i]]);
                nodes_in_element2.push_back(nodes[node_indices_element_2[i]]);
                nodes_in_element3.push_back(nodes[node_indices_element_3[i]]);
            }
            nodes_in_element4.push_back(nodes[node_indices_element_4[i]]);
        }

        std::vector<VertexElement<2,2>*> elements;
        elements.push_back(new VertexElement<2,2>(0, nodes_in_element0));
        elements.push_back(new VertexElement<2,2>(1, nodes_in_element1));
        elements.push_back(new VertexElement<2,2>(2, nodes_in_element2));
        elements.push_back(new VertexElement<2,2>(3, nodes_in_element3));
        elements.push_back(new VertexElement<2,2>(4, nodes_in_element4));

        MutableVertexMesh<2,2> mesh(nodes, elements);
        mesh.SetMeshOperationTracking(true);

        // Test default value of mCheckForT3Swaps, as well as setting and getting
        TS_ASSERT_EQUALS(mesh.GetCheckForT3Swaps(), true);
        mesh.SetCheckForT3Swaps(false);
        TS_ASSERT_EQUALS(mesh.GetCheckForT3Swaps(), false);
        mesh.SetCheckForT3Swaps(true);
        TS_ASSERT_EQUALS(mesh.GetCheckForT3Swaps(), true);

        // Set the threshold distance between vertices for a T3 swap as follows, to ease calculations
        mesh.SetCellRearrangementThreshold(0.1*1.0/1.5);

        // Node 6 is close to, but not overlapping, an edge of element 0
        TS_ASSERT_EQUALS(mesh.ElementIncludesPoint(mesh.GetNode(6)->rGetLocation(), 0), false);

        // Move node 6 to the left so that it overlaps element 1
        ChastePoint<2> point = mesh.GetNode(6)->GetPoint();
        point.SetCoordinate(0u, 0.9);
        mesh.SetNode(6, point);

        TS_ASSERT_EQUALS(mesh.ElementIncludesPoint(mesh.GetNode(6)->rGetLocation(), 0), true);
        TS_ASSERT_EQUALS(mesh.GetLocalIndexForElementEdgeClosestToPoint(mesh.GetNode(6)->rGetLocation(), 0), 1u);

        // Node 8 is close to, but not overlapping, an edge of element 0
        TS_ASSERT_EQUALS(mesh.ElementIncludesPoint(mesh.GetNode(8)->rGetLocation(), 0), false);

        // Move node 8 to the left so that it overlaps element 1
        point.SetCoordinate(0u, 0.1);
        mesh.SetNode(8, point);

        TS_ASSERT_EQUALS(mesh.ElementIncludesPoint(mesh.GetNode(8)->rGetLocation(), 0), true);
        TS_ASSERT_EQUALS(mesh.GetLocalIndexForElementEdgeClosestToPoint(mesh.GetNode(8)->rGetLocation(), 0), 3u);

        // Call method to update mesh in this situation
        mesh.ReMesh();

        // Test if the swap has been recorded properly
        auto operation_recorder = mesh.GetOperationRecorder();
        const std::vector<EdgeOperation>& edge_operations = operation_recorder->GetEdgeOperations();
        const unsigned num_operations = edge_operations.size();
        //Two node merging operations in two elements and two new edge operations in the other two elements
        TS_ASSERT_EQUALS(num_operations, 8u);
        unsigned num_edge_splits = 0;
        unsigned num_new_edges = 0;
        std::vector<std::vector<unsigned> > element_to_operations(5);
        for (unsigned i=0; i<num_operations; ++i)
        {
            if (edge_operations[i].GetOperation() == EDGE_OPERATION_SPLIT)
            {
                num_edge_splits++;
            }
            if (edge_operations[i].GetOperation() == EDGE_OPERATION_ADD)
            {
                num_new_edges++;
            }
            // Determine operations that an element underwent
            const unsigned elem_index = edge_operations[i].GetElementIndex();
            element_to_operations[elem_index].push_back(edge_operations[i].GetOperation());
        }
        TS_ASSERT_EQUALS(num_edge_splits, 5u);
        TS_ASSERT_EQUALS(num_new_edges, 3u);

        // Save the mesh data using mesh writers
        std::string dirname = "TempyTempy";
        std::string mesh_filename = "vertex_remesh_T3";
        VertexMeshWriter<2,2> mesh_writer(dirname, mesh_filename, false);
        mesh_writer.WriteFilesUsingMesh(mesh);

        // Check that node 6 has been moved onto the edge a new node has been created and both added to elements 0 amd 1
        TS_ASSERT_EQUALS(mesh.GetNumElements(), 5u);
        TS_ASSERT_EQUALS(mesh.GetNumNodes(), 16u);

        // Test locations of moved and new nodes due to node 6
        TS_ASSERT_DELTA(mesh.GetNode(6)->rGetLocation()[0], 1.0, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(6)->rGetLocation()[1], 0.5, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(13)->rGetLocation()[0], 1.0, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(13)->rGetLocation()[1], 0.4, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(14)->rGetLocation()[0], 1.0, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(14)->rGetLocation()[1], 0.6, 1e-4);

        // Test locations of moved and new nodes due to node 8
        TS_ASSERT_DELTA(mesh.GetNode(8)->rGetLocation()[0], 0.0, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(8)->rGetLocation()[1], 0.45, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(15)->rGetLocation()[0], 0.0, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(15)->rGetLocation()[1], 0.55, 1e-4);

        // Test that each element contains the correct number nodes following the rearrangement
        TS_ASSERT_EQUALS(mesh.GetElement(0)->GetNumNodes(), 9u);
        TS_ASSERT_EQUALS(mesh.GetElement(1)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(mesh.GetElement(2)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(mesh.GetElement(3)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(mesh.GetElement(4)->GetNumNodes(), 6u);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned new_node_indices_element_0[9] = {0, 1, 13, 6, 14, 2, 3, 15, 8};
        unsigned new_node_indices_element_1[4] = {4, 12, 6, 13};
        unsigned new_node_indices_element_2[4] = {12, 5, 14, 6};
        unsigned new_node_indices_element_3[4] = {7, 8, 15, 9};
        unsigned new_node_indices_element_4[6] = {10, 11, 4, 1, 0, 7};
        for (unsigned i=0; i<9; i++)
        {
            TS_ASSERT_EQUALS(mesh.GetElement(0)->GetNodeGlobalIndex(i), new_node_indices_element_0[i]);
            if (i < 4)
            {
                TS_ASSERT_EQUALS(mesh.GetElement(1)->GetNodeGlobalIndex(i), new_node_indices_element_1[i]);
                TS_ASSERT_EQUALS(mesh.GetElement(2)->GetNodeGlobalIndex(i), new_node_indices_element_2[i]);
                TS_ASSERT_EQUALS(mesh.GetElement(3)->GetNodeGlobalIndex(i), new_node_indices_element_3[i]);
            }
            if (i < 6)
            {
                // Other elements remain the same so get a void
                TS_ASSERT_EQUALS(mesh.GetElement(4)->GetNodeGlobalIndex(i), new_node_indices_element_4[i]);
            }
        }

        // Test boundary property of nodes (all are boundary nodes except node 6)
        for (unsigned i=0; i<mesh.GetNumNodes(); i++)
        {
            bool expected_boundary_node = (i!=6);
            TS_ASSERT_EQUALS(mesh.GetNode(i)->IsBoundaryNode(), expected_boundary_node);
        }

        // Test T3 swap Location tracking
        std::vector<T3SwapInfo<2> > swap_info
                            = mesh.GetOperationRecorder()->GetT3SwapsInfo();
        TS_ASSERT_EQUALS(swap_info.size(), 2u);
        std::vector< c_vector<double, 2> > t3_locations = mesh.GetLocationsOfT3Swaps();

        TS_ASSERT_DELTA(t3_locations[0][0], 1.0, 1e-6);
        TS_ASSERT_DELTA(t3_locations[0][1], 0.5, 1e-6);
        TS_ASSERT_DELTA(t3_locations[1][0], 0.0, 1e-6);
        TS_ASSERT_DELTA(t3_locations[1][1], 0.5, 1e-6);

        OutputFileHandler handler("TestT3InfoArchiving", false);
        std::string archive_filename = handler.GetOutputDirectoryFullPath() + "T3Info.arch";

        {
            // Create an output archive
            std::ofstream ofs(archive_filename.c_str());
            boost::archive::text_oarchive output_arch(ofs);

            // Write the T3 swap info
            output_arch << *mesh.GetOperationRecorder();
        }

        // Test T3 swap Location clearing
        mesh.ClearLocationsOfT3Swaps();
        swap_info = mesh.GetOperationRecorder()->GetT3SwapsInfo();
        TS_ASSERT_EQUALS(swap_info.size(), 0u);

        // Retrieve the archive
        {
            // Load T3 swap info
            std::ifstream ifs(archive_filename.c_str(), std::ios::binary);
            boost::archive::text_iarchive input_arch(ifs);

            VertexMeshOperationRecorder<2,2> recorder;
            input_arch >> recorder;
            std::vector<T3SwapInfo<2> > all_swaps = recorder.GetT3SwapsInfo();
            TS_ASSERT_EQUALS(all_swaps.size(), 2u);

            TS_ASSERT_DELTA(all_swaps[0].mLocation[0], 1.0, 1e-6);
            TS_ASSERT_DELTA(all_swaps[0].mLocation[1], 0.5, 1e-6);
            TS_ASSERT_DELTA(all_swaps[1].mLocation[0], 0.0, 1e-6);
            TS_ASSERT_DELTA(all_swaps[1].mLocation[1], 0.5, 1e-6);
        }
    }

    void TestPerformT3SwapException()
    {
        /*
         * Create a mesh comprising three joined triangular elements intersecting at a node inside a square element,
         * as shown below. We will test that trying to perform a T3 swap results in the correct exception message
         * being thrown.
         *  ______
         * |      |   /|
         * |      |  /_|
         * |      | // |
         * |      | \\_|
         * |      |  \ |
         * |______|   \|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.9, 0.5));
        nodes.push_back(new Node<2>(5, true, 2.0, 0.0));
        nodes.push_back(new Node<2>(6, true, 2.0, 0.3));
        nodes.push_back(new Node<2>(7, true, 2.0, 0.7));
        nodes.push_back(new Node<2>(8, true, 2.0, 1.0));

        std::vector<Node<2>*> nodes_in_element0, nodes_in_element1, nodes_in_element2, nodes_in_element3;
        unsigned node_indices_element_0[3] = { 1, 2, 3 };
        unsigned node_indices_element_1[3] = { 5, 6, 4 };
        unsigned node_indices_element_2[3] = { 6, 7, 4 };
        unsigned node_indices_element_3[3] = { 7, 8, 4 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_in_element0.push_back(nodes[node_indices_element_0[i]]);
            nodes_in_element1.push_back(nodes[node_indices_element_1[i]]);
            nodes_in_element2.push_back(nodes[node_indices_element_2[i]]);
            nodes_in_element3.push_back(nodes[node_indices_element_3[i]]);
        }

        std::vector<VertexElement<2, 2>*> elements;
        elements.push_back(new VertexElement<2, 2>(0, nodes_in_element0));
        elements.push_back(new VertexElement<2, 2>(1, nodes_in_element1));
        elements.push_back(new VertexElement<2, 2>(2, nodes_in_element2));
        elements.push_back(new VertexElement<2, 2>(3, nodes_in_element3));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, elements);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 9u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 4u);

        TS_ASSERT_THROWS_THIS(vertex_mesh.ReMesh(), "Trying to merge a node, contained in more than 2 elements, into another element, this is not possible with the vertex mesh.");
    }

    void TestPerformT3SwapAnotherException()
    {
        /*
         * Create a mesh comprising six nodes contained in three elements, as shown below.
         * We will move the top left node to intersect the central horizontal edge and test
         * that trying to perform a T3 swap results throws the correct exception.
         *   __
         *  |  |\
         *  |__| \
         *   \ | /
         *    \|/
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, false, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 1.0, -1.0));
        nodes.push_back(new Node<2>(5, true, 2.0, 0.0));

        std::vector<Node<2>*> nodes_in_element0, nodes_in_element1, nodes_in_element2;
        unsigned node_indices_element_0[4] = { 0, 1, 2, 3 };
        unsigned node_indices_element_1[3] = { 4, 5, 2 };
        unsigned node_indices_element_2[3] = { 0, 4, 1 };
        for (unsigned i = 0; i < 4; i++)
        {
            nodes_in_element0.push_back(nodes[node_indices_element_0[i]]);
            if (i < 3)
            {
                nodes_in_element1.push_back(nodes[node_indices_element_1[i]]);
                nodes_in_element2.push_back(nodes[node_indices_element_2[i]]);
            }
        }

        std::vector<VertexElement<2, 2>*> elements;
        elements.push_back(new VertexElement<2, 2>(0, nodes_in_element0));
        elements.push_back(new VertexElement<2, 2>(1, nodes_in_element1));
        elements.push_back(new VertexElement<2, 2>(2, nodes_in_element2));

        MutableVertexMesh<2, 2> mesh(nodes, elements);

        // Move node 3  so that it overlaps element 2 across an internal edge
        ChastePoint<2> point = mesh.GetNode(3)->GetPoint();
        point.SetCoordinate(0, 0.5);
        point.SetCoordinate(1, -0.1);
        mesh.SetNode(3, point);

        TS_ASSERT_DELTA(mesh.GetNode(3)->rGetLocation()[0], 0.5, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(3)->rGetLocation()[1], -0.1, 1e-4);

        TS_ASSERT_EQUALS(mesh.ElementIncludesPoint(mesh.GetNode(3)->rGetLocation(), 2), true);
        TS_ASSERT_EQUALS(mesh.GetLocalIndexForElementEdgeClosestToPoint(mesh.GetNode(3)->rGetLocation(), 0), 2u);

        TS_ASSERT_THROWS_THIS(mesh.ReMesh(), "A boundary node has intersected a non-boundary edge; this is because the boundary element has become concave. You need to rerun the simulation with a smaller time step to prevent this.");
    }

    void TestT3SwapOnSmallEdge()
    {
        /*
         * Create a mesh comprising eight nodes contained in one rectangle and two
         * triangle elements, as shown below. We will test intersecting on an edge
         * that is too small.
         *
         *  ______  /|
         * |      |/_| <---
         * |______|\ |
         *          \|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 0.1));
        nodes.push_back(new Node<2>(3, true, 0.0, 0.1));
        nodes.push_back(new Node<2>(4, true, 0.99, 0.05));
        nodes.push_back(new Node<2>(5, true, 2.0, 0.0));
        nodes.push_back(new Node<2>(6, true, 2.0, 0.1));
        nodes.push_back(new Node<2>(7, true, 2.0, 0.0));

        std::vector<Node<2>*> nodes_in_element0, nodes_in_element1, nodes_in_element2;
        unsigned node_indices_element_0[4] = { 0, 1, 2, 3 };
        unsigned node_indices_element_1[3] = { 5, 6, 4 };
        unsigned node_indices_element_2[3] = { 7, 4, 6 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_in_element0.push_back(nodes[node_indices_element_0[i]]);
            nodes_in_element1.push_back(nodes[node_indices_element_1[i]]);
            nodes_in_element2.push_back(nodes[node_indices_element_2[i]]);
        }
        nodes_in_element0.push_back(nodes[node_indices_element_0[3]]);

        std::vector<VertexElement<2, 2>*> elements;
        elements.push_back(new VertexElement<2, 2>(0, nodes_in_element0));
        elements.push_back(new VertexElement<2, 2>(1, nodes_in_element1));
        elements.push_back(new VertexElement<2, 2>(2, nodes_in_element2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, elements);

        // Set the threshold distance between vertices for a T3 swap as follows, to ease calculations
        vertex_mesh.SetCellRearrangementThreshold(0.1/1.5);
        vertex_mesh.SetMeshOperationTracking(true);//For coverage
        vertex_mesh.SetCellRearrangementThreshold(0.1 / 1.5);
        vertex_mesh.SetCellRearrangementThreshold(0.1 / 1.5);

        // Call PerformT3Swap(); note that we don't call ReMesh(), since this would also perform T1 swaps
        vertex_mesh.PerformT3Swap(vertex_mesh.GetNode(4), 0u);

        TS_ASSERT_EQUALS(Warnings::Instance()->GetNextWarningMessage(), "Trying to merge a node onto an edge which is too small.");
        Warnings::QuietDestroy();

        // Check that node 4 has been moved onto the edge and a new node has been created and both added to elements 0 and 1
        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 10u);

        // Test locations of moved and new nodes
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 1.00, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.05, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(8)->rGetLocation()[0], 1.00, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(8)->rGetLocation()[1], -0.05, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(9)->rGetLocation()[0], 1.00, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(9)->rGetLocation()[1], 0.15, 1e-8);

        // Test locations of edges of intersected edge
        TS_ASSERT_DELTA(vertex_mesh.GetNode(1)->rGetLocation()[0], 1.00, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(1)->rGetLocation()[1], -0.15, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(2)->rGetLocation()[0], 1.00, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(2)->rGetLocation()[1], 0.25, 1e-8);

        // Test that each element contains the correct number of nodes following the rearrangement
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 7u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 4u);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned new_node_indices_element_0[7] = { 0, 1, 8, 4, 9, 2, 3 };
        unsigned new_node_indices_element_1[4] = { 5, 6, 4, 8 };
        unsigned new_node_indices_element_2[4] = { 7, 9, 4, 6 };
        for (unsigned i = 0; i < 7; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), new_node_indices_element_0[i]);
            if (i < 4)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), new_node_indices_element_1[i]);
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), new_node_indices_element_2[i]);
            }
        }

        // Test boundary property of nodes
        for (unsigned i = 0; i < vertex_mesh.GetNumNodes(); i++)
        {
            bool expected_boundary_node = (i != 4);
            TS_ASSERT_EQUALS(vertex_mesh.GetNode(i)->IsBoundaryNode(), expected_boundary_node);
        }
    }

    void TestConsecutiveT3SwapsForSmallEdges()
    {
        /*
         * We create a mesh like this:
         *
         *        0     /\
         *   \         /  \            /
         *    \_______|____\__________/
         *            |     \
         *            |  1   \
         *
         * Two nodes that are very close to each other overlap with element 0.
         * After the first T3 swap we expect a situation like this:
         *
         *                C
         *                |\
         *   \            | \          /    (Note: in the test below the image is
         *    \___________|__\________/            flipped, i.e. Node B would be on the left.)
         *         A|     |   B
         *          |      \
         *
         * A further T3 Swap would merge node C onto edge AB, which is an edge of
         * its own element. We prevent this by just removing node C.  By doing this we
         * also avoid element 1 to be concave.
         *
         * After deleting Node C we should end up with something like this,
         *
         *   \                         /
         *    \_______________B_______/
         *         A|          \
         *          |           \
         *
         * i.e. the two overlapping nodes were effectively merged onto the closest edge and the
         * new vertices are only slightly more than the cell rearrangement threshold distance apart.
         */

        // We start by creating nodes and from them a mesh.
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 2.0, 1.0));
        nodes.push_back(new Node<2>(3, true, -1.0, 1.0));
        nodes.push_back(new Node<2>(4, true, -2.0, -2.0));
        nodes.push_back(new Node<2>(5, true, 2.0, -2.0));
        nodes.push_back(new Node<2>(6, true, 0.55, 0.11));
        nodes.push_back(new Node<2>(7, true, 0.5, 0.01));

        std::vector<Node<2>*> nodes_in_element0, nodes_in_element1;
        unsigned node_indices_element_0[4] = { 0, 1, 2, 3 };
        unsigned node_indices_element_1[4] = { 4, 5, 6, 7 };
        for (unsigned i = 0; i < 4; i++)
        {
            nodes_in_element0.push_back(nodes[node_indices_element_0[i]]);
            nodes_in_element1.push_back(nodes[node_indices_element_1[i]]);
        }

        std::vector<VertexElement<2, 2>*> elements;
        elements.push_back(new VertexElement<2, 2>(0, nodes_in_element0));
        elements.push_back(new VertexElement<2, 2>(1, nodes_in_element1));

        MutableVertexMesh<2,2> vertex_mesh(nodes, elements);
        vertex_mesh.SetMeshOperationTracking(true);//For coverage

        // Set the threshold distance between vertices for a T3 swap as follows, to ease calculations.
        vertex_mesh.SetCellRearrangementThreshold(0.2);

        // Node 6 and 7 are overlapping an edge of element 0
        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(6)->rGetLocation(), 0), true);
        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(7)->rGetLocation(), 0), true);

        // Check for and perform intersections - should return true!
        TS_ASSERT(vertex_mesh.CheckForIntersections());

        // Check that node 6 has been moved onto the edge and a new node has been created and added to both elements
        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 9u);

        // Node 6 now has 2 elements whereas node 7 has only one element
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(6)->GetNumContainingElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(7)->GetNumContainingElements(), 1u);

        // Element 0 should have 6 nodes and element 1 should have 5 nodes
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 6u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 5u);

        // Node 7 should be intersecting element 0
        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(7)->rGetLocation(), 0), true);

        // Node 8 should be the starting point of the edge that 7 is closest to. This means that the point would be merged
        // back onto the freshly created edge from the previous swap if the ReMesh method doesn't check whether this is happening.
        unsigned edge_closest_to_7_local_index = vertex_mesh.GetLocalIndexForElementEdgeClosestToPoint(vertex_mesh.GetNode(7)->rGetLocation(), 0);
        unsigned edge_closest_to_7_global_index = vertex_mesh.GetElement(0)->GetNode(edge_closest_to_7_local_index)->GetIndex();
        TS_ASSERT_EQUALS(edge_closest_to_7_global_index, 8u);

        c_vector<double, 2> location_node_6_before_swap;
        location_node_6_before_swap = vertex_mesh.GetNode(6)->rGetLocation();
        c_vector<double, 2> location_node_8_before_swap;
        location_node_8_before_swap = vertex_mesh.GetNode(8)->rGetLocation();

        // We perform the next swap:
        TS_ASSERT(vertex_mesh.CheckForIntersections());

        // We should now have lost one node, i.e. having 8 nodes now (node 7 should be marked as deleted)
        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 8u);

        // Element 0 should have 6 nodes and element 1 should have 4 nodes
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 6u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 4u);

        // The locations of node 6 and 8 should not be changed!
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(6)->rGetLocation()[0], location_node_6_before_swap[0]);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(6)->rGetLocation()[1], location_node_6_before_swap[1]);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(8)->rGetLocation()[0], location_node_8_before_swap[0]);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(8)->rGetLocation()[1], location_node_8_before_swap[1]);

        // The two elements should have 2 nodes in common and they should both be boundary nodes
        unsigned num_common_vertices = 0;
        for (unsigned i = 0; i < vertex_mesh.GetElement(0)->GetNumNodes(); i++)
        {
            for (unsigned j = 0; j < vertex_mesh.GetElement(1)->GetNumNodes(); j++)
            {
                if ((vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i)) == (vertex_mesh.GetElement(1)->GetNodeGlobalIndex(j)))
                {
                    num_common_vertices++;
                    TS_ASSERT(vertex_mesh.GetNode(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i))->IsBoundaryNode());
                }
            }
        }
        TS_ASSERT_EQUALS(num_common_vertices, 2u);
    }

    void TestT3SwapForNeighbouringElements()
    {
        /*
         * Create a mesh comprising nine nodes contained in one square and three triangle elements,
         * as shown below. We will test that a T3 swap is performed correctly in this case.
         *         _____
         *        |     |
         *     /\ |     | /|\
         *    /__\|_____|/_|_\
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 1.5, 0.0));
        nodes.push_back(new Node<2>(5, true, 2.0, 0.0));
        nodes.push_back(new Node<2>(6, true, 1.5, 0.5));
        nodes.push_back(new Node<2>(7, true, -1.0, 0.0));
        nodes.push_back(new Node<2>(8, true, -0.5, 0.5));

        std::vector<Node<2>*> nodes_in_element0, nodes_in_element1, nodes_in_element2, nodes_in_element3;
        unsigned node_indices_element_0[4] = { 0, 1, 2, 3 };
        unsigned node_indices_element_1[3] = { 1, 4, 6 };
        unsigned node_indices_element_2[3] = { 4, 5, 6 };
        unsigned node_indices_element_3[3] = { 7, 0, 8 };
        for (unsigned i = 0; i < 4; i++)
        {
            nodes_in_element0.push_back(nodes[node_indices_element_0[i]]);
            if (i < 3)
            {
                nodes_in_element1.push_back(nodes[node_indices_element_1[i]]);
                nodes_in_element2.push_back(nodes[node_indices_element_2[i]]);
                nodes_in_element3.push_back(nodes[node_indices_element_3[i]]);
            }
        }

        std::vector<VertexElement<2, 2>*> elements;
        elements.push_back(new VertexElement<2, 2>(0, nodes_in_element0));
        elements.push_back(new VertexElement<2, 2>(1, nodes_in_element1));
        elements.push_back(new VertexElement<2, 2>(2, nodes_in_element2));
        elements.push_back(new VertexElement<2, 2>(3, nodes_in_element3));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, elements);

        // Set the threshold distance between vertices for a T3 swap as follows, to ease calculations
        vertex_mesh.SetCellRearrangementThreshold(0.1*1.0/1.5);
        vertex_mesh.SetMeshOperationTracking(true);//For coverage
        vertex_mesh.SetCellRearrangementThreshold(0.1 * 1.0 / 1.5);
        vertex_mesh.SetCellRearrangementThreshold(0.1 * 1.0 / 1.5);

        // Node 6 and 8 are close to, but not overlapping, an edge of element 0
        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(6)->rGetLocation(), 0), false);
        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(8)->rGetLocation(), 0), false);

        // Move node 6 to the left so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(6)->GetPoint();
        point.SetCoordinate(0u, 0.9);
        vertex_mesh.SetNode(6, point);

        TS_ASSERT_DELTA(vertex_mesh.GetNode(6)->rGetLocation()[0], 0.9, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(6)->rGetLocation()[1], 0.5, 1e-4);

        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(6)->rGetLocation(), 0), true);
        TS_ASSERT_EQUALS(vertex_mesh.GetLocalIndexForElementEdgeClosestToPoint(vertex_mesh.GetNode(6)->rGetLocation(), 0), 1u);

        // Move node 8 to the right so that it overlaps element 0
        point.SetCoordinate(0u, 0.1);
        vertex_mesh.SetNode(8, point);

        TS_ASSERT_DELTA(vertex_mesh.GetNode(8)->rGetLocation()[0], 0.1, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(8)->rGetLocation()[1], 0.5, 1e-4);

        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(8)->rGetLocation(), 0), true);
        TS_ASSERT_EQUALS(vertex_mesh.GetLocalIndexForElementEdgeClosestToPoint(vertex_mesh.GetNode(8)->rGetLocation(), 0), 3u);

        // Call method to update vertex_mesh in this situation
        vertex_mesh.ReMesh(); //MoveOverlappingNodeOntoEdgeOfElement(vertex_mesh.GetNode(6), 0);

        // Check that node 6 has been moved onto the edge a new node has been created and both added to elements 0 amd 1
        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 10u);

        // Test locations of moved and new nodes
        TS_ASSERT_DELTA(vertex_mesh.GetNode(6)->rGetLocation()[0], 1.0, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(6)->rGetLocation()[1], 0.45, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(9)->rGetLocation()[0], 1.0, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(9)->rGetLocation()[1], 0.55, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(8)->rGetLocation()[0], 0.0, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(8)->rGetLocation()[1], 0.5, 1e-4);

        // Test that each element contains the correct number of nodes following the rearrangement
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 7u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNumNodes(), 3u);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned new_node_indices_element_0[7] = { 0, 1, 6, 9, 2, 3, 8 };
        unsigned new_node_indices_element_1[3] = { 1, 4, 6 };
        unsigned new_node_indices_element_2[4] = { 4, 5, 9, 6 };
        unsigned new_node_indices_element_3[3] = { 7, 0, 8 };
        for (unsigned i = 0; i < 7; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), new_node_indices_element_0[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), new_node_indices_element_1[i]);
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNodeGlobalIndex(i), new_node_indices_element_3[i]);
            }
            if (i < 4)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), new_node_indices_element_2[i]);
            }
        }

        // Test boundary property of nodes. All are boundary nodes except node 6.
        for (unsigned i = 0; i < vertex_mesh.GetNumNodes(); i++)
        {
            bool expected_boundary_node = (i != 6);
            TS_ASSERT_EQUALS(vertex_mesh.GetNode(i)->IsBoundaryNode(), expected_boundary_node);
        }
    }

    void TestResolveTriangularOverlapAfterConsecutiveT3Swaps()
    {
        /*
         *  Create a mesh with an overlap as shown below.
         *
         *  \           1             /
         *   \ A____________B        /
         *    /              \      /
         *   / \______________\____/
         *  /  C       0       \
         *
         *  The overlapping node A will be merged onto the left slanted edge,
         *  whereas the overlapping B node will be merged onto the lower horizontal edge.
         *  This leaves node C only belonging to element 1 but not to element 2 and overlapping
         *  element one, like this
         *
         *  \      1       /
         *  E\____________/F
         *   /    \/      \
         *  /      C  0    \
         *
         * In this situation we would like the mesh to connect nodes E and F and delete all nodes in between
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.1));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.1));
        nodes.push_back(new Node<2>(2, true, 2.0, -2.0));
        nodes.push_back(new Node<2>(3, true, -2.0, -2.0));
        nodes.push_back(new Node<2>(4, true, -2.0, 2.0));
        nodes.push_back(new Node<2>(5, true, 2.0, 2.0));
        nodes.push_back(new Node<2>(6, true, 2.0, 0.0));
        nodes.push_back(new Node<2>(7, true, 0.0, 0.0));

        std::vector<Node<2>*> nodes_in_element0, nodes_in_element1;
        unsigned node_indices_element_0[4] = { 0, 3, 2, 1 };
        unsigned node_indices_element_1[4] = { 7, 6, 5, 4 };
        for (unsigned i = 0; i < 4; i++)
        {
            nodes_in_element0.push_back(nodes[node_indices_element_0[i]]);
            nodes_in_element1.push_back(nodes[node_indices_element_1[i]]);
        }

        std::vector<VertexElement<2, 2>*> elements;
        elements.push_back(new VertexElement<2, 2>(0, nodes_in_element0));
        elements.push_back(new VertexElement<2, 2>(1, nodes_in_element1));

        MutableVertexMesh<2,2> vertex_mesh(nodes, elements);
        vertex_mesh.SetMeshOperationTracking(true);//For coverage

        // Set the threshold distance between vertices for a T3 swap as follows, to ease calculations
        vertex_mesh.SetCellRearrangementThreshold(0.15);

        // Node 0 and 1 are overlapping an edge of element 0
        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(0)->rGetLocation(), 1), true);
        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(1)->rGetLocation(), 1), true);

        // Check for and perform intersections - should return true!
        TS_ASSERT(vertex_mesh.CheckForIntersections());

        // Check that node 0 has been moved onto the edge a new node has been created and added to both elements
        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 9u);

        // Node 0 now has 2 elements whereas nodes 1 and 7 have only one element
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(0)->GetNumContainingElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(1)->GetNumContainingElements(), 1u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(7)->GetNumContainingElements(), 1u);

        // We perform the next swap:
        TS_ASSERT(vertex_mesh.CheckForIntersections());

        // We made another node
        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 10u);

        // and now node 1 also has two elements whereas node 7 still doesn't
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(0)->GetNumContainingElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(1)->GetNumContainingElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(7)->GetNumContainingElements(), 1u);

        // The two elements should have 4 nodes in common and all the common nodes are boundary nodes
        unsigned num_common_vertices = 0;
        for (unsigned i = 0; i < vertex_mesh.GetElement(0)->GetNumNodes(); i++)
        {
            for (unsigned j = 0; j < vertex_mesh.GetElement(1)->GetNumNodes(); j++)
            {
                if ((vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i)) == (vertex_mesh.GetElement(1)->GetNodeGlobalIndex(j)))
                {
                    num_common_vertices++;
                    TS_ASSERT(vertex_mesh.GetNode(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i))->IsBoundaryNode());
                }
            }
        }
        TS_ASSERT_EQUALS(num_common_vertices, 4u);

        // Perform the next swap
        TS_ASSERT(vertex_mesh.CheckForIntersections());

        // The mesh should now have 7 nodes and the other nodes should be deleted
        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 2u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 7u);

        // The two elements should now only have 2 nodes in common and all the common nodes are boundary nodes
        num_common_vertices = 0;
        for (unsigned i = 0; i < vertex_mesh.GetElement(0)->GetNumNodes(); i++)
        {
            for (unsigned j = 0; j < vertex_mesh.GetElement(1)->GetNumNodes(); j++)
            {
                if ((vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i)) == (vertex_mesh.GetElement(1)->GetNodeGlobalIndex(j)))
                {
                    num_common_vertices++;
                    TS_ASSERT(vertex_mesh.GetNode(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i))->IsBoundaryNode());
                }
            }
        }
        TS_ASSERT_EQUALS(num_common_vertices, 2u);
    }

    void TestT3SwapForNeighbouringElementsWithTwoCommonNodes()
    {
        /*
         * Create a small mesh as shown below.
         *        _____
         *       |     |
         *     /\|     |/|\
         *    /__|_____|_|_\
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.00));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.00));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.00));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.00));
        nodes.push_back(new Node<2>(4, true, 1.5, 0.00));
        nodes.push_back(new Node<2>(5, true, 2.0, 0.00));
        nodes.push_back(new Node<2>(6, true, 1.5, 0.50));
        nodes.push_back(new Node<2>(7, true, -1.0, 0.00));
        nodes.push_back(new Node<2>(8, true, -0.5, 0.50));
        nodes.push_back(new Node<2>(9, true, 1.0, 0.25));
        nodes.push_back(new Node<2>(10, true, 0.0, 0.25));

        std::vector<Node<2>*> nodes_in_element0, nodes_in_element1, nodes_in_element2, nodes_in_element3;
        unsigned node_indices_element_0[6] = { 0, 1, 9, 2, 3, 10 };
        unsigned node_indices_element_1[4] = { 1, 4, 6, 9 };
        unsigned node_indices_element_2[3] = { 4, 5, 6 };
        unsigned node_indices_element_3[4] = { 7, 0, 10, 8 };
        for (unsigned i = 0; i < 6; i++)
        {
            nodes_in_element0.push_back(nodes[node_indices_element_0[i]]);
            if (i < 4)
            {
                nodes_in_element1.push_back(nodes[node_indices_element_1[i]]);
                nodes_in_element3.push_back(nodes[node_indices_element_3[i]]);
            }
            if (i < 3)
            {
                nodes_in_element2.push_back(nodes[node_indices_element_2[i]]);
            }
        }

        std::vector<VertexElement<2, 2>*> elements;
        elements.push_back(new VertexElement<2, 2>(0, nodes_in_element0));
        elements.push_back(new VertexElement<2, 2>(1, nodes_in_element1));
        elements.push_back(new VertexElement<2, 2>(2, nodes_in_element2));
        elements.push_back(new VertexElement<2, 2>(3, nodes_in_element3));

        MutableVertexMesh<2,2> vertex_mesh(nodes, elements);
        vertex_mesh.SetMeshOperationTracking(true);//For coverage

        // Set the threshold distance between vertices for a T3 swap as follows, to ease calculations
        vertex_mesh.SetCellRearrangementThreshold(0.1 * 1.0 / 1.5);

        // Node 6 and 8 are close to, but not overlapping, an edge of element 0
        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(6)->rGetLocation(), 0), false);
        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(8)->rGetLocation(), 0), false);

        // Move node 6 to the left so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(6)->GetPoint();
        point.SetCoordinate(0u, 0.9);
        vertex_mesh.SetNode(6, point);

        TS_ASSERT_DELTA(vertex_mesh.GetNode(6)->rGetLocation()[0], 0.9, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(6)->rGetLocation()[1], 0.5, 1e-4);

        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(6)->rGetLocation(), 0), true);
        TS_ASSERT_EQUALS(vertex_mesh.GetLocalIndexForElementEdgeClosestToPoint(vertex_mesh.GetNode(6)->rGetLocation(), 0), 2u);

        // Move node 8 to the right so that it overlaps element 0
        point.SetCoordinate(0u, 0.1);
        vertex_mesh.SetNode(8, point);

        TS_ASSERT_DELTA(vertex_mesh.GetNode(8)->rGetLocation()[0], 0.1, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(8)->rGetLocation()[1], 0.5, 1e-4);

        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(8)->rGetLocation(), 0), true);
        TS_ASSERT_EQUALS(vertex_mesh.GetLocalIndexForElementEdgeClosestToPoint(vertex_mesh.GetNode(8)->rGetLocation(), 0), 4u);

        // Call method to update vertex_mesh in this situation
        vertex_mesh.ReMesh(); //MoveOverlappingNodeOntoEdgeOfElement(vertex_mesh.GetNode(6), 0);

        // Check that node 6 has been moved onto the edge a new node has been created and both added to elements 0 and 1
        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 10u);

        // Test locations of moved and new nodes (9 is the next free node when 6 is merged)
        TS_ASSERT_DELTA(vertex_mesh.GetNode(6)->rGetLocation()[0], 1.00, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(6)->rGetLocation()[1], 0.45, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(9)->rGetLocation()[0], 1.00, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(9)->rGetLocation()[1], 0.55, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(8)->rGetLocation()[0], 0.00, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(8)->rGetLocation()[1], 0.50, 1e-4);

        // Test that each element contains the correct number of nodes following the rearrangement
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 7u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNumNodes(), 3u);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned new_node_indices_element_0[7] = { 0, 1, 6, 9, 2, 3, 8 };
        unsigned new_node_indices_element_1[3] = { 1, 4, 6 };
        unsigned new_node_indices_element_2[4] = { 4, 5, 9, 6 };
        unsigned new_node_indices_element_3[3] = { 7, 0, 8 };
        for (unsigned i = 0; i < 7; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), new_node_indices_element_0[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), new_node_indices_element_1[i]);
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNodeGlobalIndex(i), new_node_indices_element_3[i]);
            }
            if (i < 4)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), new_node_indices_element_2[i]);
            }
        }

        // Test boundary property of nodes (all are boundary nodes except node 6)
        for (unsigned i = 0; i < vertex_mesh.GetNumNodes(); i++)
        {
            bool expected_boundary_node = (i != 6);
            TS_ASSERT_EQUALS(vertex_mesh.GetNode(i)->IsBoundaryNode(), expected_boundary_node);
        }
    }

    void TestReMeshForT3Swaps()
    {
        /**
         * Load a vertex mesh from file, in which several nodes are intersecting other elements.
         * We will test that T3 swaps are correctly performed to resolve these intersections.
         *     ____
         *    _\ | /_
         * |\|  \|/  |/|
         * | \       /_|
         * | /       \ |
         * |/|__/\___|\|
         *     /__\
         *
         *      |\                /|
         *      |_\ |          | /_|
         *      | / v          v \ |
         *  ____|/________________\|
         *  \  /|                  |
         *   \/ |                  |
         *   -->|                  |<--
         *      |                  | /\
         *      |__________________|/__\
         *      |\                /|
         *      |_\ ^          ^ /_|
         *      | / |          | \ |
         *      |/                \|
         *
         *      |\  |          |  /|
         *      |_\ v          v /_|
         *   ___|_/______________\_|
         *   \  |                  |
         *    \/|                  |
         *   -->|                  |<--
         *      |                  |/\
         *      |__________________|__\
         *      |_\              /_|
         *      | / ^         ^  \ |
         *      |/  |         |   \|
         */
        VertexMeshReader<2, 2> mesh_reader("cell_based/test/data/TestMutableVertexMesh/vertex_remesh_T3");
        MutableVertexMesh<2, 2> vertex_mesh;
        vertex_mesh.ConstructFromMeshReader(mesh_reader);

        vertex_mesh.SetMeshOperationTracking(true);//For coverage
        vertex_mesh.SetDistanceForT3SwapChecking(100.0);
        // Set the threshold distance between vertices for a T3 swap as follows, to ease calculations
        vertex_mesh.SetCellRearrangementThreshold(0.1 * 1.0 / 1.5);

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 29u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 64u);

        // Calls ReMesh() to identify and perform any T3 swaps (element overlaps)
        vertex_mesh.ReMesh();

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 29u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 72u);

        // Save the mesh data using mesh writers
        std::string dirname = "TestVertexMeshReMesh";
        std::string mesh_filename = "vertex_remesh_T3";
        VertexMeshWriter<2, 2> mesh_writer(dirname, mesh_filename, false);
        mesh_writer.WriteFilesUsingMesh(vertex_mesh);

        // Check the positions are updated correctly
        OutputFileHandler handler("TestVertexMeshReMesh", false);
        std::string results_file1 = handler.GetOutputDirectoryFullPath() + "vertex_remesh_T3.node";
        std::string results_file2 = handler.GetOutputDirectoryFullPath() + "vertex_remesh_T3.cell";

        FileComparison comparer1(results_file1, "cell_based/test/data/TestMutableVertexMesh/vertex_remesh_T3_after_remesh.node");
        TS_ASSERT(comparer1.CompareFiles());
        FileComparison comparer2(results_file2, "cell_based/test/data/TestMutableVertexMesh/vertex_remesh_T3_after_remesh.cell");
        TS_ASSERT(comparer2.CompareFiles());
    }

    void TestReMeshForRemovingVoids()
    {
        /*
         * Create a mesh comprising eight nodes contained in three elements with a small central void,
         * as shown below. We will test that the void is correctly removed from the mesh by a T2 swap.
         *  ______       _______
         * |     /|     |      /|
         * |___/| |     |_____/ |
         * |   \| | --> |     \ |
         * |_____\|     |______\|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.00, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.00, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.00, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.00, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.40, 0.5));
        nodes.push_back(new Node<2>(5, true, 0.55, 0.4));
        nodes.push_back(new Node<2>(6, true, 0.55, 0.6));
        nodes.push_back(new Node<2>(7, true, 0.00, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[5] = { 0, 1, 5, 4, 7 };
        unsigned node_indices_elem_1[4] = { 1, 2, 6, 5 };
        unsigned node_indices_elem_2[5] = { 7, 4, 6, 2, 3 };
        for (unsigned i = 0; i < 5; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            if (i < 4)
            {
                nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            }
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2,2> vertex_mesh(nodes, vertex_elements);
        vertex_mesh.SetMeshOperationTracking(true);//For coverage

        vertex_mesh.SetCellRearrangementThreshold(0.1);

        vertex_mesh.ReMesh(); // Edges too long so nothing happens

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 8u);

        c_vector<double, 2>& new_location_0 = vertex_mesh.GetNode(5)->rGetModifiableLocation();
        new_location_0(1) = 0.51;

        c_vector<double, 2>& new_location_1 = vertex_mesh.GetNode(6)->rGetModifiableLocation();
        new_location_1(1) = 0.49;

        // T1 swap should now happen, removing the void
        vertex_mesh.ReMesh();

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 6u);

        // Test merged node is in the correct place
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.5, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.5, 1e-4);

        // Test ownership of the new nodes
        std::set<unsigned> expected_elements_containing_node_4;
        expected_elements_containing_node_4.insert(0);
        expected_elements_containing_node_4.insert(1);
        expected_elements_containing_node_4.insert(2);

        TS_ASSERT_EQUALS(vertex_mesh.GetNode(4)->rGetContainingElementIndices(), expected_elements_containing_node_4);

        // Test that each element contains the correct number nodes following the rearrangement
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 4u);

        // Test that each element contains the correct nodes following the rearrangement
        // (note that the node indices were reset since the void was removed)
        unsigned node_indices_element_0[4] = { 0, 1, 4, 5 };
        unsigned node_indices_element_1[3] = { 1, 2, 4 };
        unsigned node_indices_element_2[4] = { 5, 4, 2, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            }
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
        }
    }

    void TestReMeshForRemovingVoidsException()
    {
        /*
         * Create a mesh comprising eleven nodes contain in four elements with two small
         * triangle voids, as shown below. We will test that trying to remove the voids
         * results in the correct exception being thrown.
         *    _________
         *   |        /|
         *   |/|\___/| |
         *   |\|/   \| |
         *   |________\|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.00, 0.00));
        nodes.push_back(new Node<2>(1, true, 1.00, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.00, 1.00));
        nodes.push_back(new Node<2>(3, true, 0.00, 1.00));
        nodes.push_back(new Node<2>(4, true, 0.45, 0.49));
        nodes.push_back(new Node<2>(5, true, 0.60, 0.50));
        nodes.push_back(new Node<2>(6, true, 0.45, 0.51));
        nodes.push_back(new Node<2>(7, true, 0.80, 0.50));
        nodes.push_back(new Node<2>(8, true, 0.90, 0.51));
        nodes.push_back(new Node<2>(9, true, 0.90, 0.49));
        nodes.push_back(new Node<2>(10, true, 1.00, 0.50));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3;

        unsigned node_indices_elem_0[7] = { 0, 1, 10, 9, 7, 5, 4 };
        unsigned node_indices_elem_1[4] = { 0, 4, 6, 3 };
        unsigned node_indices_elem_2[7] = { 7, 8, 10, 2, 3, 6, 5 };
        unsigned node_indices_elem_3[3] = { 10, 8, 9 };

        for (unsigned i = 0; i < 7; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            if (i < 4)
            {
                nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            }
            if (i < 3)
            {
                nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
            }
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));

        MutableVertexMesh<2,2> vertex_mesh(nodes, vertex_elements);
        vertex_mesh.SetMeshOperationTracking(true);//For coverage

        vertex_mesh.SetCellRearrangementThreshold(0.1);

        // Call IdentifySwapType on nodes 6 and 4 (ordering for coverage)
        vertex_mesh.IdentifySwapType(vertex_mesh.GetNode(6), vertex_mesh.GetNode(4));

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 9u);

        // Test merged node is in the correct place
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.5, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.5, 1e-4);

        // Test ownership of the new nodes
        std::set<unsigned> expected_elements_containing_node_4;
        expected_elements_containing_node_4.insert(0);
        expected_elements_containing_node_4.insert(1);
        expected_elements_containing_node_4.insert(2);

        TS_ASSERT_EQUALS(vertex_mesh.GetNode(4)->rGetContainingElementIndices(), expected_elements_containing_node_4);

        // Call IdentifySwapType() on nodes 6 and 7 (originally nodes 8 and 9)
        TS_ASSERT_THROWS_THIS(vertex_mesh.IdentifySwapType(vertex_mesh.GetNode(6), vertex_mesh.GetNode(7)),
                              "Triangular element next to triangular void, not implemented yet.");
    }

    void TestT3SwapForRemovingVoids()
    {
        /*
         * Create a mesh comprising seven nodes containing in three elements with a central
         * void, as shown below. We will test that a T3 swap is correctly performed to
         * remove the void from the mesh.
         *  ______       _______
         * |     /|     |      /|
         * |___/| |     |_____/ |
         * |   \| | --> |     \ |
         * |_____\|     |______\|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.4, 0.5));
        nodes.push_back(new Node<2>(5, true, 0.5, 0.4));
        nodes.push_back(new Node<2>(6, true, 0.5, 0.6));
        nodes.push_back(new Node<2>(7, true, 0.0, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[5] = { 0, 1, 5, 4, 7 };
        unsigned node_indices_elem_1[4] = { 1, 2, 6, 5 };
        unsigned node_indices_elem_2[5] = { 7, 4, 6, 2, 3 };
        for (unsigned i = 0; i < 5; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            if (i < 4)
            {
                nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            }
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2,2> vertex_mesh(nodes, vertex_elements);
        vertex_mesh.SetMeshOperationTracking(true);//For coverage

        vertex_mesh.ReMesh(); // Edges too long so nothing happens

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 8u);

        // Move node 4 to inside of element 1
        c_vector<double, 2>& new_location = vertex_mesh.GetNode(4)->rGetModifiableLocation();
        new_location(0) = 0.6;

        TS_ASSERT_EQUALS(vertex_mesh.ElementIncludesPoint(vertex_mesh.GetNode(4)->rGetLocation(), 1), true);
        TS_ASSERT_EQUALS(vertex_mesh.GetLocalIndexForElementEdgeClosestToPoint(vertex_mesh.GetNode(4)->rGetLocation(), 1), 2u);

        // T3 swap should now happen, removing the void
        vertex_mesh.ReMesh();

        TS_ASSERT_EQUALS(vertex_mesh.GetNumElements(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetNumNodes(), 6u);

        // Test merged node is in the correct place
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.5, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.5, 1e-4);

        // Test ownership of the new nodes
        std::set<unsigned> expected_elements_containing_node_4;
        expected_elements_containing_node_4.insert(0);
        expected_elements_containing_node_4.insert(1);
        expected_elements_containing_node_4.insert(2);

        TS_ASSERT_EQUALS(vertex_mesh.GetNode(4)->rGetContainingElementIndices(), expected_elements_containing_node_4);

        // Test that each element contains the correct number of nodes following the rearrangement
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 4u);

        // Test that each element contains the correct nodes following the rearrangement
        // (note that the node indices have been reset since the void was removed)
        unsigned node_indices_element_0[4] = { 0, 1, 4, 5 };
        unsigned node_indices_element_1[3] = { 1, 2, 4 };
        unsigned node_indices_element_2[4] = { 5, 4, 2, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            }
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
        }

        // Test that the correct nodes are boundary nodes
        for (unsigned i=0; i<vertex_mesh.GetNumNodes(); i++)
        {
            bool expected_boundary_node = (i!=4);
            TS_ASSERT_EQUALS(vertex_mesh.GetNode(i)->IsBoundaryNode(), expected_boundary_node);
        }
    }

    void TestT3SwapWithConcaveElements()
    {
        /*
         * Create a mesh comprising ten nodes contained in one rectangle element,
         * one L-shape element and one small triangle element. We will test that
         * a T3 swap correctly removes the triangle element in this case.
         *     ______       _______
         *    |____|/      |_ _ | /
         *    | |____  --> | |_\|/_
         *    |______|     |_______|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 2.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 2.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 1.0, 2.0));
        nodes.push_back(new Node<2>(5, true, 0.0, 2.0));
        nodes.push_back(new Node<2>(6, true, 2.0, 2.0));
        nodes.push_back(new Node<2>(7, true, 2.0, 3.0));
        nodes.push_back(new Node<2>(8, true, 0.0, 3.0));
        nodes.push_back(new Node<2>(9, true, 3.0, 3.0));

        std::vector<Node<2>*> nodes_in_element0, nodes_in_element1, nodes_in_element2;
        unsigned node_indices_in_element0[6] = { 0, 1, 2, 3, 4, 5 };
        unsigned node_indices_in_element1[5] = { 5, 4, 6, 7, 8 };
        unsigned node_indices_in_element2[3] = { 7, 6, 9 };
        for (unsigned i = 0; i < 6; i++)
        {
            nodes_in_element0.push_back(nodes[node_indices_in_element0[i]]);
            if (i < 5)
            {
                nodes_in_element1.push_back(nodes[node_indices_in_element1[i]]);
            }
            if (i < 3)
            {
                nodes_in_element2.push_back(nodes[node_indices_in_element2[i]]);
            }
        }

        std::vector<VertexElement<2, 2>*> elements;
        elements.push_back(new VertexElement<2, 2>(0, nodes_in_element0));
        elements.push_back(new VertexElement<2, 2>(1, nodes_in_element1));
        elements.push_back(new VertexElement<2, 2>(2, nodes_in_element2));

        MutableVertexMesh<2,2> mesh(nodes, elements);
        mesh.SetMeshOperationTracking(true);//For coverage

        // Set the threshold distance between vertices for a T3 swap as follows, to ease calculations
        mesh.SetCellRearrangementThreshold(0.1 * 1.0 / 1.5);

        // Move node 6  so that it overlaps element 0
        ChastePoint<2> point = mesh.GetNode(6)->GetPoint();
        point.SetCoordinate(0u, 1.5);
        point.SetCoordinate(1u, 0.9);
        mesh.SetNode(6, point);

        TS_ASSERT_DELTA(mesh.GetNode(6)->rGetLocation()[0], 1.5, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(6)->rGetLocation()[1], 0.9, 1e-4);

        TS_ASSERT_EQUALS(mesh.ElementIncludesPoint(mesh.GetNode(6)->rGetLocation(), 0), true);
        TS_ASSERT_EQUALS(mesh.GetLocalIndexForElementEdgeClosestToPoint(mesh.GetNode(6)->rGetLocation(), 0), 2u);

        // Call method to update mesh in this situation
        mesh.ReMesh(); //MoveOverlappingNodeOntoEdgeOfElement(mesh.GetNode(6), 0);

        // Check that node 6 has been moved onto the edge a new node has been created and both added to elements 0 and 1
        TS_ASSERT_EQUALS(mesh.GetNumElements(), 3u);
        TS_ASSERT_EQUALS(mesh.GetNumNodes(), 12u);

        // Test locations of moved and new nodes (10 and 11 are the next free node when 6 is merged)
        TS_ASSERT_DELTA(mesh.GetNode(6)->rGetLocation()[0], 1.5, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(6)->rGetLocation()[1], 1.0, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(10)->rGetLocation()[0], 1.6, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(10)->rGetLocation()[1], 1.0, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(11)->rGetLocation()[0], 1.4, 1e-4);
        TS_ASSERT_DELTA(mesh.GetNode(11)->rGetLocation()[1], 1.0, 1e-4);

        // Test that each element contains the number of correct nodes following the rearrangement
        TS_ASSERT_EQUALS(mesh.GetElement(0)->GetNumNodes(), 9u);
        TS_ASSERT_EQUALS(mesh.GetElement(1)->GetNumNodes(), 6u);
        TS_ASSERT_EQUALS(mesh.GetElement(2)->GetNumNodes(), 4u);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[9] = { 0, 1, 2, 10, 6, 11, 3, 4, 5 };
        unsigned node_indices_element_1[6] = { 5, 4, 11, 6, 7, 8 };
        unsigned node_indices_element_2[4] = { 7, 6, 10, 9 };
        for (unsigned i = 0; i < 9; i++)
        {
            TS_ASSERT_EQUALS(mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            if (i < 6)
            {
                TS_ASSERT_EQUALS(mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            }
            if (i < 4)
            {
                TS_ASSERT_EQUALS(mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            }
        }

        // Test that the correct nodes are boundary nodes
        for (unsigned i = 0; i < mesh.GetNumNodes(); i++)
        {
            bool expected_boundary_node = (i != 6);
            TS_ASSERT_EQUALS(mesh.GetNode(i)->IsBoundaryNode(), expected_boundary_node);
        }
    }

    void TestPerformIntersectionSwap()
    {
        /*
         * Create a mesh comprising six nodes contained in two triangle and two
         * rhomboid elements, as follows:
         *
         *   _____________
         *  |\           /|
         *  |  \    0   / |
         *  |    \     /  |
         *  |      \  /   |
         *  | 3   ---   1 |
         *  |   /  \      |
         *  |  /     \    |
         *  | /   2    \  |
         *  |/___________\|
         *
         * We will test that when a node is moved to overlap with an element, it is correctly
         * found and dealt with by the CheckForIntersections() method.
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, false, 0.4, 0.5));
        nodes.push_back(new Node<2>(5, false, 0.6, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3;
        unsigned node_indices_elem_0[3] = { 2, 3, 5 };
        unsigned node_indices_elem_1[4] = { 2, 5, 4, 1 };
        unsigned node_indices_elem_2[3] = { 1, 4, 0 };
        unsigned node_indices_elem_3[4] = { 0, 4, 5, 3 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);
        nodes_elem_3.push_back(nodes[node_indices_elem_3[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));

        MutableVertexMesh<2,2> vertex_mesh(nodes, vertex_elements);
        vertex_mesh.SetMeshOperationTracking(true);//For coverage

        // Move node 4 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(4)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(4, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        vertex_mesh.CheckForIntersections();

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.7, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.5, 1e-3);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 2, 3, 4, 5 };
        unsigned node_indices_element_1[3] = { 2, 5, 1 };
        unsigned node_indices_element_2[4] = { 1, 5, 4, 0 };
        unsigned node_indices_element_3[3] = { 0, 4, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNodeGlobalIndex(i), node_indices_element_3[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.36, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(3), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.2806, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 2.7294, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(3), 2.3062, 1e-4);
    }

    void TestPerformIntersectionSwapOtherWayRound()
    {
        /*
         * This test is very similar to TestPerformIntersectionSwap() but with a different ordering
         * of nodes and elements, to ensure full coverage of the CheckForIntersections() method.
         *
         * The mesh is as follows:
         *
         *   _____________
         *  |\           /|
         *  | \    0   /  |
         *  |  \     /    |
         *  |   \  /      |
         *  | 1   ---   3 |
         *  |      /  \   |
         *  |    /     \  |
         *  |  /   2    \ |
         *  |/___________\|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, false, 0.4, 0.5));
        nodes.push_back(new Node<2>(5, false, 0.6, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3;
        unsigned node_indices_elem_0[3] = { 2, 3, 4 };
        unsigned node_indices_elem_1[4] = { 0, 5, 4, 3 };
        unsigned node_indices_elem_2[3] = { 1, 5, 0 };
        unsigned node_indices_elem_3[4] = { 2, 4, 5, 1 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);
        nodes_elem_3.push_back(nodes[node_indices_elem_3[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));

        MutableVertexMesh<2,2> vertex_mesh(nodes, vertex_elements);
        vertex_mesh.SetMeshOperationTracking(true);//For coverage

        // Move node 5 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(5)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(5, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        vertex_mesh.CheckForIntersections();

        // Test the rearrangement is correct, as in TestPerformIntersectionSwap()
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.5, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.7, 1e-3);

        unsigned node_indices_element_0[4] = { 2, 3, 4, 5 };
        unsigned node_indices_element_1[3] = { 0, 4, 3 };
        unsigned node_indices_element_2[4] = { 1, 5, 4, 0 };
        unsigned node_indices_element_3[3] = { 2, 5, 1 };
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNodeGlobalIndex(i), node_indices_element_3[i]);
            }
        }

        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.36, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(3), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.2806, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 2.7294, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(3), 2.3062, 1e-4);
    }

    void TestPerformIntersectionSwapOtherIndexing()
    {
        /*
         * This test is very similar to TestPerformIntersectionSwap() but with a different indexing
         * of nodes, to ensure that any problems with local indexing are caught.
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, false, 0.4, 0.5));
        nodes.push_back(new Node<2>(5, false, 0.6, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3;
        unsigned node_indices_elem_0[3] = { 5, 2, 3 };
        unsigned node_indices_elem_1[4] = { 2, 5, 4, 1 };
        unsigned node_indices_elem_2[3] = { 4, 0, 1 };
        unsigned node_indices_elem_3[4] = { 0, 4, 5, 3 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);
        nodes_elem_3.push_back(nodes[node_indices_elem_3[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 4 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(4)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(4, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        vertex_mesh.CheckForIntersections();

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.7, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.5, 1e-3);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 5, 2, 3, 4 };
        unsigned node_indices_element_1[3] = { 2, 5, 1 };
        unsigned node_indices_element_2[4] = { 4, 0, 1, 5 };
        unsigned node_indices_element_3[3] = { 0, 4, 3 };
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(3)->GetNodeGlobalIndex(i), node_indices_element_3[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.36, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(3), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.2806, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 2.7294, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(3), 2.3062, 1e-4);
    }

    void TestPerformIntersectionSwapTriangle()
    {
        /*
         * This test is very similar to TestPerformIntersectionSwap() but the intersected
         * element is a triangle element, to ensure that this exception is caught.
         *
         * The mesh is as follows:
         *
         *   _____________
         *   \           /|
         *    \\    0   / |
         *     \3\     /  |
         *      \  \  /   |
         *        ---   1 |
         *      /  \      |
         *     /     \    |
         *    /   2    \  |
         *   /___________\|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.4, 0.5));
        nodes.push_back(new Node<2>(5, false, 0.6, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3;
        unsigned node_indices_elem_0[3] = { 2, 3, 5 };
        unsigned node_indices_elem_1[4] = { 2, 5, 4, 1 };
        unsigned node_indices_elem_2[3] = { 1, 4, 0 };
        unsigned node_indices_elem_3[4] = { 4, 5, 3 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 4 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(4)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(4, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        TS_ASSERT_THROWS_THIS(vertex_mesh.CheckForIntersections(), "A triangular element has become concave. "
                                                                   "You need to rerun the simulation with a smaller time step to prevent this.");
    }

    void TestPerformIntersectionSwapTriangleOtherWayRound()
    {
        /*
         * This test is very similar to TestPerformIntersectionSwapTriangle() but the intersected
         * element is a triangle element, to ensure that this exception is caught.
         *
         * The mesh is as follows:
         *
         *   _____________
         *  |\           /
         *  | \    0   //
         *  |  \     /3/
         *  |   \  /  /
         *  | 1   ---
         *  |      /  \
         *  |    /     \
         *  |  /   2    \
         *  |/___________\
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, false, 0.4, 0.5));
        nodes.push_back(new Node<2>(5, true, 0.6, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3;
        unsigned node_indices_elem_0[3] = { 2, 3, 4 };
        unsigned node_indices_elem_1[4] = { 0, 5, 4, 3 };
        unsigned node_indices_elem_2[3] = { 1, 5, 0 };
        unsigned node_indices_elem_3[4] = { 2, 4, 5 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 5 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(5)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(5, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_THROWS_THIS(vertex_mesh.CheckForIntersections(), "A triangular element has become concave. "
                                                                   "You need to rerun the simulation with a smaller time step to prevent this.");
    }

    void TestPerformIntersectionSwapSplitting()
    {
        /*
         * Create a mesh comprising eight nodes contained in two triangle, two rhomboid elements,
         * and one pentagonal element.  We will test that an exception is raised when an intersection
         * splits the element into two new elements.
         *
         * The mesh is as follows:
         *
         *   _____________
         *  |\          //|
         *  |   \   0 / / |
         *  |      \/ 1/  |
         *  |       | /   |
         *  | 4   ---   2 |
         *  |   /  \      |
         *  |  /     \    |
         *  | /   3    \  |
         *  |/___________\|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.8, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(5, false, 0.4, 0.5));
        nodes.push_back(new Node<2>(6, false, 0.6, 0.5));
        nodes.push_back(new Node<2>(7, false, 0.6, 0.6));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2, nodes_elem_3, nodes_elem_4;
        unsigned node_indices_elem_0[3] = { 3, 4, 7 };
        unsigned node_indices_elem_1[4] = { 2, 3, 7, 6 };
        unsigned node_indices_elem_2[4] = { 1, 2, 6, 5 };
        unsigned node_indices_elem_3[3] = { 0, 1, 5 };
        unsigned node_indices_elem_4[5] = { 0, 5, 6, 7, 4 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_3.push_back(nodes[node_indices_elem_3[i]]);
        }
        for (unsigned i = 0; i < 4; i++)
        {
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
        }
        for (unsigned i = 0; i < 5; i++)
        {
            nodes_elem_4.push_back(nodes[node_indices_elem_4[i]]);
        }

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));
        vertex_elements.push_back(new VertexElement<2, 2>(3, nodes_elem_3));
        vertex_elements.push_back(new VertexElement<2, 2>(4, nodes_elem_4));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 5 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(5)->GetPoint();
        point.SetCoordinate(1u, 0.8);
        vertex_mesh.SetNode(5, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_THROWS_THIS(vertex_mesh.CheckForIntersections(), "Intersection cannot be resolved "
                                                                   "without splitting the element into two new elements.");
    }

    void TestPerformBoundaryIntersectionSwap1()
    {
        /*
         * This test is similar to TestPerformIntersectionSwap, but with the
         * bottom triangle element missing, such that the intersecting node is
         * a boundary node.
         *
         * The mesh is as follows:
         *
         *   _____________
         *  |\           /|
         *  |  \    0   / |
         *  |    \     /  |
         *  |      \  /   |
         *  | 2   ---   1 |
         *  |   /  \      |
         *  |  /     \    |
         *  | /        \  |
         *  |/           \|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.4, 0.5));
        nodes.push_back(new Node<2>(5, false, 0.6, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[3] = { 2, 3, 5 };
        unsigned node_indices_elem_1[4] = { 2, 5, 4, 1 };
        unsigned node_indices_elem_2[4] = { 0, 4, 5, 3 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);
        nodes_elem_2.push_back(nodes[node_indices_elem_2[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 4 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(4)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(4, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        vertex_mesh.CheckForIntersections();

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.7, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.5, 1e-3);

        // Test that each moved node is boundary node
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(4)->IsBoundaryNode(), true);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(5)->IsBoundaryNode(), true);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 2, 3, 4, 5 };
        unsigned node_indices_element_1[3] = { 2, 5, 1 };
        unsigned node_indices_element_2[3] = { 0, 4, 3 };
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 3u);
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.2806, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 2.3062, 1e-4);
    }

    void TestPerformBoundaryIntersectionSwapOtherWayRound1()
    {
        /*
         * This test is similar to TestPerformBoundaryIntersectionSwap1, but
         * with a different ordering of nodes and elements, to ensure full
         * coverage of the CheckForIntersections() method.
         *
         * The mesh is as follows:
         *
         *   _____________
         *  |\           /|
         *  | \    0   /  |
         *  |  \     /    |
         *  |   \  /      |
         *  | 1   ---   2 |
         *  |      /  \   |
         *  |    /     \  |
         *  |  /        \ |
         *  |/           \|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, false, 0.4, 0.5));
        nodes.push_back(new Node<2>(5, true, 0.6, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[3] = { 2, 3, 4 };
        unsigned node_indices_elem_1[4] = { 0, 5, 4, 3 };
        unsigned node_indices_elem_2[4] = { 2, 4, 5, 1 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);
        nodes_elem_2.push_back(nodes[node_indices_elem_2[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 4 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(5)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(5, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        vertex_mesh.CheckForIntersections();

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.5, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.7, 1e-3);

        // Test that each moved node is boundary node
        TS_ASSERT(vertex_mesh.GetNode(4)->IsBoundaryNode());
        TS_ASSERT(vertex_mesh.GetNode(5)->IsBoundaryNode());

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 2, 3, 4, 5 };
        unsigned node_indices_element_1[3] = { 0, 4, 3 };
        unsigned node_indices_element_2[3] = { 2, 5, 1 };
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 3u);
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.2806, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 2.3062, 1e-4);
    }

    void TestPerformBoundaryIntersectionSwap2()
    {
        /*
         * This test is similar to TestPerformIntersectionSwap, but with the
         * right rhomboid element missing, such that the intersecting node is a
         * boundary node.
         *
         * The mesh is as follows:
         *
         *   _____________
         *  |\           /
         *  |  \    0   /
         *  |    \     /
         *  |      \  /
         *  | 1   ---
         *  |   /  \
         *  |  /     \
         *  | /   2    \
         *  |/___________\
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.4, 0.5));
        nodes.push_back(new Node<2>(5, true, 0.6, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[3] = { 2, 3, 5 };
        unsigned node_indices_elem_1[4] = { 0, 4, 5, 3 };
        unsigned node_indices_elem_2[3] = { 1, 4, 0 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 4 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(4)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(4, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        vertex_mesh.CheckForIntersections();

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.7, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.5, 1e-3);

        // Test that each moved node is boundary node
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(4)->IsBoundaryNode(), false);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(5)->IsBoundaryNode(), true);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 2, 3, 4, 5 };
        unsigned node_indices_element_1[3] = { 0, 4, 3 };
        unsigned node_indices_element_2[4] = { 1, 5, 4, 0 };
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 4u);
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.36, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.3062, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 2.7294, 1e-4);
    }

    void TestPerformBoundaryIntersectionSwapOtherWayRound2()
    {
        /*
         * This test is similar to TestPerformBoundaryIntersectionSwap2, but
         * with a different ordering of nodes and elements, to ensure full
         * coverage of the CheckForIntersections() method.
         *
         * The mesh is as follows:
         *
         *   _____________
         *  |\           /
         *  | \    0   /
         *  |  \     /
         *  |   \  /
         *  | 1   ---
         *  |      /  \
         *  |    /     \
         *  |  /   2    \
         *  |/___________\
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.4, 0.5));
        nodes.push_back(new Node<2>(5, true, 0.6, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[3] = { 2, 3, 4 };
        unsigned node_indices_elem_1[4] = { 0, 5, 4, 3 };
        unsigned node_indices_elem_2[3] = { 1, 5, 0 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 4 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(5)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(5, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        vertex_mesh.CheckForIntersections();

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.5, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.7, 1e-3);

        // Test that each moved node is boundary node
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(4)->IsBoundaryNode(), false);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(5)->IsBoundaryNode(), true);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 2, 3, 4, 5 };
        unsigned node_indices_element_1[3] = { 0, 4, 3 };
        unsigned node_indices_element_2[4] = { 1, 5, 4, 0 };
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 4u);
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.36, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.2806, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 2.7294, 1e-4);
    }

    void TestPerformBoundaryIntersectionSwap3()
    {
        /*
         * This test is similar to TestPerformIntersectionSwap, but with the
         * left rhomboid element missing, such that the intersecting node is a
         * boundary node.
         *
         * The mesh is as follows:
         *
         *   _____________
         *   \           /|
         *     \    0   / |
         *       \     /  |
         *         \  /   |
         *        ---   1 |
         *      /  \      |
         *     /     \    |
         *    /   2    \  |
         *   /___________\|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.4, 0.5));
        nodes.push_back(new Node<2>(5, true, 0.6, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[3] = { 2, 3, 5 };
        unsigned node_indices_elem_1[4] = { 2, 5, 4, 1 };
        unsigned node_indices_elem_2[3] = { 1, 4, 0 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 4 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(4)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(4, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        vertex_mesh.CheckForIntersections();

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.7, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.5, 1e-3);

        // Test that each moved node is boundary node
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(4)->IsBoundaryNode(), true);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(5)->IsBoundaryNode(), false);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 2, 3, 4, 5 };
        unsigned node_indices_element_1[3] = { 2, 5, 1 };
        unsigned node_indices_element_2[4] = { 1, 5, 4, 0 };
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 4u);
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.36, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.2806, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 2.7294, 1e-4);
    }

    void TestPerformBoundaryIntersectionSwapOtherWayRound3()
    {
        /*
         * This test is similar to TestPerformBoundaryIntersectionSwap3, but
         * with a different ordering of nodes and elements, to ensure full
         * coverage of the CheckForIntersections() method.
         *
         * The mesh is as follows:
         *
         *   _____________
         *   \           /|
         *    \    0   /  |
         *     \     /    |
         *      \  /      |
         *        ---   1 |
         *         /  \   |
         *       /     \  |
         *     /   2    \ |
         *   /___________\|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(2, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.4, 0.5));
        nodes.push_back(new Node<2>(5, true, 0.6, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1, nodes_elem_2;
        unsigned node_indices_elem_0[3] = { 2, 3, 4 };
        unsigned node_indices_elem_1[4] = { 2, 4, 5, 1 };
        unsigned node_indices_elem_2[3] = { 1, 5, 0 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_2.push_back(nodes[node_indices_elem_2[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));
        vertex_elements.push_back(new VertexElement<2, 2>(2, nodes_elem_2));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 4 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(5)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(5, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        vertex_mesh.CheckForIntersections();

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.5, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(5)->rGetLocation()[1], 0.7, 1e-3);

        // Test that each moved node is boundary node
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(4)->IsBoundaryNode(), true);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(5)->IsBoundaryNode(), false);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 2, 3, 4, 5 };
        unsigned node_indices_element_1[3] = { 2, 5, 1 };
        unsigned node_indices_element_2[4] = { 1, 5, 4, 0 };
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNumNodes(), 4u);
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(2)->GetNodeGlobalIndex(i), node_indices_element_2[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(2), 0.36, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.3062, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(2), 2.7294, 1e-4);
    }

    void TestPerformBoundaryIntersectionSwap4()
    {
        /*
         * This test is similar to TestPerformIntersectionSwap, but with the
         * bottom triangle and right rhomboid element missing, such that the
         * intersecting node is a boundary node.
         *
         * The mesh is as follows:
         *
         *   _____________
         *  |\           /
         *  |  \    0   /
         *  |    \     /
         *  |      \  /
         *  | 1   ---
         *  |   /
         *  |  /
         *  | /
         *  |/
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 0.4, 0.5));
        nodes.push_back(new Node<2>(2, true, 0.6, 0.5));
        nodes.push_back(new Node<2>(3, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.0, 1.0));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1;
        unsigned node_indices_elem_0[3] = { 3, 4, 2 };
        unsigned node_indices_elem_1[4] = { 0, 1, 2, 4 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 1 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(1)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(1, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        vertex_mesh.CheckForIntersections();

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(1)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(1)->rGetLocation()[1], 0.7, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(2)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(2)->rGetLocation()[1], 0.5, 1e-3);

        // Test that each moved node is boundary node
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(1)->IsBoundaryNode(), true);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(2)->IsBoundaryNode(), true);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 3, 4, 1, 2 };
        unsigned node_indices_element_1[3] = { 0, 1, 4 };
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.3062, 1e-4);
    }

    void TestPerformBoundaryIntersectionSwapOtherWayRound4()
    {
        /*
         * This test is similar to TestPerformBoundaryIntersectionSwap4, but
         * with a different ordering of nodes and elements, to ensure full
         * coverage of the CheckForIntersections() method.
         *
         * The mesh is as follows:
         *
         *   _____________
         *  |\           /
         *  | \    0   /
         *  |  \     /
         *  |   \  /
         *  | 1   ---
         *  |      /
         *  |    /
         *  |  /
         *  |/
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 0.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 0.4, 0.5));
        nodes.push_back(new Node<2>(2, true, 0.6, 0.5));
        nodes.push_back(new Node<2>(3, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(4, true, 0.0, 1.0));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1;
        unsigned node_indices_elem_0[3] = { 3, 4, 1 };
        unsigned node_indices_elem_1[4] = { 0, 2, 1, 4 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 1 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(2)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(2, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        vertex_mesh.CheckForIntersections();

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(1)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(1)->rGetLocation()[1], 0.5, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(2)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(2)->rGetLocation()[1], 0.7, 1e-3);

        // Test that each moved node is boundary node
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(1)->IsBoundaryNode(), true);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(2)->IsBoundaryNode(), true);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 3, 4, 1, 2 };
        unsigned node_indices_element_1[3] = { 0, 1, 4 };
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.2806, 1e-4);
    }

    void TestPerformBoundaryIntersectionSwap5()
    {
        /*
         * This test is similar to TestPerformIntersectionSwap, but with the
         * bottom triangle and left rhomboid element missing, such that the
         * intersecting node is a boundary node.
         *
         * The mesh is as follows:
         *
         *   _____________
         *   \           /|
         *     \    0   / |
         *       \     /  |
         *         \  /   |
         *        ---   1 |
         *         \      |
         *           \    |
         *             \  |
         *               \|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(2, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.6, 0.5));
        nodes.push_back(new Node<2>(4, true, 0.4, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1;
        unsigned node_indices_elem_0[3] = { 1, 2, 3 };
        unsigned node_indices_elem_1[4] = { 0, 1, 3, 4 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 4 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(4)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(4, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        vertex_mesh.CheckForIntersections();

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.7, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(3)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(3)->rGetLocation()[1], 0.5, 1e-3);

        // Test that each moved node is boundary node
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(4)->IsBoundaryNode(), true);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(3)->IsBoundaryNode(), true);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 1, 2, 4, 3 };
        unsigned node_indices_element_1[3] = { 0, 1, 3 };
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.2806, 1e-4);
    }

    void TestPerformBoundaryIntersectionSwapOtherWayRound5()
    {
        /*
         * This test is similar to TestPerformBoundaryIntersectionSwap5, but
         * with a different ordering of nodes and elements, to ensure full
         * coverage of the CheckForIntersections() method.
         *
         * The mesh is as follows:
         *
         *   _____________
         *   \           /|
         *    \    0   /  |
         *     \     /    |
         *      \  /      |
         *        ---   1 |
         *            \   |
         *             \  |
         *              \ |
         *               \|
         */
        std::vector<Node<2>*> nodes;
        nodes.push_back(new Node<2>(0, true, 1.0, 0.0));
        nodes.push_back(new Node<2>(1, true, 1.0, 1.0));
        nodes.push_back(new Node<2>(2, true, 0.0, 1.0));
        nodes.push_back(new Node<2>(3, true, 0.6, 0.5));
        nodes.push_back(new Node<2>(4, true, 0.4, 0.5));

        std::vector<Node<2>*> nodes_elem_0, nodes_elem_1;
        unsigned node_indices_elem_0[3] = { 1, 2, 4 };
        unsigned node_indices_elem_1[4] = { 0, 1, 4, 3 };
        for (unsigned i = 0; i < 3; i++)
        {
            nodes_elem_0.push_back(nodes[node_indices_elem_0[i]]);
            nodes_elem_1.push_back(nodes[node_indices_elem_1[i]]);
        }
        nodes_elem_1.push_back(nodes[node_indices_elem_1[3]]);

        std::vector<VertexElement<2, 2>*> vertex_elements;
        vertex_elements.push_back(new VertexElement<2, 2>(0, nodes_elem_0));
        vertex_elements.push_back(new VertexElement<2, 2>(1, nodes_elem_1));

        MutableVertexMesh<2, 2> vertex_mesh(nodes, vertex_elements);

        // Move node 3 so that it overlaps element 0
        ChastePoint<2> point = vertex_mesh.GetNode(3)->GetPoint();
        point.SetCoordinate(1u, 0.7);
        vertex_mesh.SetNode(3, point);

        // Merge intersection to maintain non-overlapping elements
        vertex_mesh.SetCheckForInternalIntersections(true);
        TS_ASSERT_EQUALS(vertex_mesh.GetCheckForInternalIntersections(), true);
        vertex_mesh.CheckForIntersections();

        // Test that each moved node has the correct location following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[0], 0.4, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(4)->rGetLocation()[1], 0.5, 1e-8);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(3)->rGetLocation()[0], 0.6, 1e-3);
        TS_ASSERT_DELTA(vertex_mesh.GetNode(3)->rGetLocation()[1], 0.7, 1e-3);

        // Test that each moved node is boundary node
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(4)->IsBoundaryNode(), true);
        TS_ASSERT_EQUALS(vertex_mesh.GetNode(3)->IsBoundaryNode(), true);

        // Test that each element contains the correct nodes following the rearrangement
        unsigned node_indices_element_0[4] = { 1, 2, 4, 3 };
        unsigned node_indices_element_1[3] = { 0, 1, 3 };
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNumNodes(), 4u);
        TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNumNodes(), 3u);
        for (unsigned i = 0; i < 4; i++)
        {
            TS_ASSERT_EQUALS(vertex_mesh.GetElement(0)->GetNodeGlobalIndex(i), node_indices_element_0[i]);
            if (i < 3)
            {
                TS_ASSERT_EQUALS(vertex_mesh.GetElement(1)->GetNodeGlobalIndex(i), node_indices_element_1[i]);
            }
        }

        // Test that each element has the correct area and perimeter following the rearrangement
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(0), 0.24, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetVolumeOfElement(1), 0.20, 1e-6);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(0), 2.4232, 1e-4);
        TS_ASSERT_DELTA(vertex_mesh.GetSurfaceAreaOfElement(1), 2.3062, 1e-4);
    }
};

#endif /*TESTMUTABLEVERTEXMESHREMESH_HPP_*/
