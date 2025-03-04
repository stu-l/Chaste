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
#ifndef VERTEXMESHWRITER_HPP_
#define VERTEXMESHWRITER_HPP_

// Forward declaration prevents circular include chain
template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
class VertexMesh;

#ifdef CHASTE_VTK
//Requires  "sudo aptitude install libvtk5-dev" or similar
#define _BACKWARD_BACKWARD_WARNING_H 1 //Cut out the strstream deprecated warning for now (gcc4.3)
#include <vtkDoubleArray.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkConvexPointSet.h>
#include <vtkPolygon.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkDataCompressor.h>
#endif //CHASTE_VTK

#include "VertexMesh.hpp"
#include "AbstractMeshWriter.hpp"
#include "NodeMap.hpp"

// Forward declaration prevents circular include chain
template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
class VertexMesh;

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
struct MeshWriterIterators;

/**
 * A mesh writer class for vertex-based meshes.
 */
template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
class VertexMeshWriter : public AbstractMeshWriter<ELEMENT_DIM, SPACE_DIM>
{
private:
    /**
     * If writing from a mesh object, the mesh to write to disk.
     * Otherwise NULL.
     */
    VertexMesh<ELEMENT_DIM,SPACE_DIM>* mpMesh;

    /** Iterators over the mesh */
    MeshWriterIterators<ELEMENT_DIM,SPACE_DIM>* mpIters;

    /** Track deleted nodes so they don't get written */
    NodeMap* mpNodeMap;
    /** What was the last index written to #mpNodeMap ? */
    unsigned mNodeMapCurrentIndex;

#ifdef CHASTE_VTK
//Requires  "sudo aptitude install libvtk5-dev" or similar
///\todo Merge into VtkMeshWriter (#1076)
    vtkUnstructuredGrid* mpVtkUnstructedMesh;
#endif //CHASTE_VTK

public:

    /**
     * Constructor.
     *
     * @param rDirectory reference to the output directory, relative to where Chaste output is stored
     * @param rBaseName reference to the base name for results files
     * @param clearOutputDir whether to clear the output directory prior to writing files
     */
    VertexMeshWriter(const std::string& rDirectory,
                     const std::string& rBaseName,
                     const bool clearOutputDir=true);

    /**
     * Destructor.
     */
    ~VertexMeshWriter();

    ///\todo Mesh should be const (#1076)
    /**
     * Write files using a mesh.
     *
     * @param rMesh reference to the vertex-based mesh
     */
    void WriteFilesUsingMesh(VertexMesh<ELEMENT_DIM, SPACE_DIM>& rMesh);

    /**
     * Write VTK file using a mesh.
     *
     * @param rMesh reference to the vertex-based mesh
     * @param stamp is an optional stamp (like a time-stamp) to put into the name of the file
     */
    void WriteVtkUsingMesh(VertexMesh<ELEMENT_DIM, SPACE_DIM>& rMesh, std::string stamp="");

    /**
     * Populate mpVtkUnstructedMesh using a vertex-based mesh.
     * Called by WriteVtkUsingMesh().
     *
     * @param rMesh reference to the vertex-based mesh
     */
    void MakeVtkMesh(VertexMesh<ELEMENT_DIM, SPACE_DIM>& rMesh);

    /**
     * Add data to a future VTK file.
     *
     * @param dataName a tag to go into the VTK file
     * @param dataPayload a pay-load of length (number of elements)
     */
    void AddCellData(std::string dataName, std::vector<double> dataPayload);

    /**
     * Add data to a future VTK file.
     *
     * @param dataName a tag to go into the VTK file
     * @param dataPayload a pay-load of length (number of nodes)
     */
    void AddPointData(std::string dataName, std::vector<double> dataPayload);

    /**
     * @return the coordinates of the next node to be written to file
     */
    std::vector<double> GetNextNode();

    /**
     * @return the data (indices/attributes) of the next element to be written to file
     */
    ElementData GetNextElement();

    /**
     * @return the data (indices/attributes) of the next element to be written to file, including its faces.
     *         This method should only be called in 3D.
     */
    VertexElementData GetNextElementWithFaces();

    /**
     * Write mesh data to files.
     * This method must be overridden in concrete classes.
     */
    void WriteFiles();
};

#endif /*VERTEXMESHWRITER_HPP_*/
