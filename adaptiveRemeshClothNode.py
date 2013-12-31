"""

#
# COMSE 6998-2 Final Project: Adaptive Remeshing Cloth Simulation
# ---------------------------------------------------------------
#
# Code by Dale Zhao (dz2242)
# --------------------------
#


# To set up a test scene, first make sure that this dependency node is loaded
# using the plug-in manager. Then paste the following code into Maya's Python
# script editor:


import maya.cmds as cmds
import maya.mel as mel

def setupAaptiveRemeshClothScene():

    # Create a new scene.
    cmds.file(f = True, new = True)
    
    # Reload this plugin.
    cmds.unloadPlugin("adaptiveRemeshClothNode")
    cmds.loadPlugin("adaptiveRemeshClothNode")

    # Create a plane mesh with specified size and divisions and triangulate it.
    
    size = 5
    ndivs = 3
    cmds.polyPlane(w = size, h = size, sx = ndivs, sy = ndivs, ax = (0, 1, 0), cuv = 2, ch = True)
    cmds.polyTriangulate("pPlaneShape1", ch = True)
    
    # Create an instance of the adaptive remeshing cloth node and connect
    # the plane mesh to it as input cloth geometry.
    
    cmds.createNode("adaptiveRemeshClothNode")
    cmds.createNode("mesh", n = "arClothShape1")
    cmds.hide("pPlaneShape1")

    cmds.connectAttr("pPlaneShape1.worldMesh[0]", "adaptiveRemeshClothNode1.inputMesh")
    cmds.connectAttr("adaptiveRemeshClothNode1.outputMesh", "arClothShape1.inMesh")
    cmds.connectAttr("time1.outTime", "adaptiveRemeshClothNode1.currentTime")
    
    # Display mesh component IDs for debugging purposes.
    # cmds.ToggleVertIDs()
    # cmds.ToggleEdgeIDs()
    # cmds.ToggleFaceIDs()


setupAaptiveRemeshClothScene()


# Various simulation parameters are defined in the '__init__' method of the
# 'AdaptiveRemeshClothNode'. But after making a change, this dependency node
# has to be reloaded to take effect.


"""


import sys
import os
import math
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMayaFX as OpenMayaFX


# An ad hoc class with static methods doing linear algebra calculations.

class M:
    
    # Add two matrices.
    
    @classmethod
    def addm(cls, m1, m2):
        
        r = len(m1)
        c = len(m1[0])
        
        mout = [None] * r
        for i in range(0, r):
            mout[i] = [0.0] * c
            for j in range(0, c):
                mout[i][j] = m1[i][j] + m2[i][j]
        
        return mout
    
    
    # Subtract a matrix from another.
    
    @classmethod
    def subm(cls, m1, m2):
        
        r = len(m1)
        c = len(m1[0])
        
        mout = [None] * r
        for i in range(0, r):
            mout[i] = [0.0] * c
            for j in range(0, c):
                mout[i][j] = m1[i][j] - m2[i][j]
        
        return mout
    
    
    # Multiply two matrices.
    
    @classmethod
    def multmm(cls, m1, m2):
        
        r = len(m1)
        q = len(m1[0])
        c = len(m2[0])
        
        mout = [None] * r
        for i in range(0, r):
            mout[i] = [0.0] * c
            for j in range(0, c):
                for k in range(0, q):
                    mout[i][j] += m1[i][k] * m2[k][j]
        
        return mout
    
    
    # Multiply a matrix by a scalar.
    
    @classmethod
    def multms(cls, m1, s1):
        
        r = len(m1)
        c = len(m1[0])
        
        mout = [None] * r
        for i in range(0, r):
            mout[i] = [0.0] * c
            for j in range(0, c):
                mout[i][j] = m1[i][j] * s1
        
        return mout
    
    
    # Divide a matrix by a scalar.
    
    @classmethod
    def divms(cls, m1, s1):
        
        if s1 == 0:
            return M.copy(m1)
        else:
            return cls.multms(m1, 1.0 / s1)
    
    
    # Create a row matrix from a scalar list.
    
    @classmethod
    def row(cls, l):
        
        c = len(l)
        mout = [[0.0] * c]
        for i in range(0, c):
            mout[0][i] = l[i]
        
        return mout
    
    
    # Create a column matrix from a scalar list.
    
    @classmethod
    def column(cls, l):
        
        r = len(l)
        mout = [None] * r
        for i in range(0, r):
            mout[i] = [l[i]]
        
        return mout
    
    # Create a zero matrix.
    
    @classmethod
    def zero(cls, r, c):
        
        mout = [None] * r
        for i in range(0, r):
            mout[i] = [0.0] * c
            
        return mout
    
    
    # Create a diagonal matrix from a scalar list.
    
    @classmethod
    def diag(cls, l):
        
        r = len(l)
        
        mout = [None] * r
        for i in range(0, r):
            mout[i] = [0.0] * r
            mout[i][i] = l[i]
        
        return mout
    
    
    # Transpose a matrix.
    
    @classmethod
    def transposem(cls, m1):
        
        r = len(m1)
        c = len(m1[0])
        
        mout = [None] * c
        for i in range(0, c):
            mout[i] = [0.0] * r
            for j in range(0, r):
                mout[i][j] = m1[j][i]
        
        return mout
    
    
    # Compute the maximum norm of a matrix.
    
    @classmethod
    def maxNorm(cls, m1):
        
        return max(map(max, M.abs(m1)))
    
    
    # Compute the square of the Frobenius norm (2-norm) of a matrix.
    
    @classmethod
    def frobNorm2(cls, m1):
        
        s = 0.0
        
        for i in range(0, len(m1)):
            for j in range(0, len(m1[0])):
                e = m1[i][j]
                s += e * e
        
        return s
    
    
    # Compute the Frobenius norm (2-norm) of a matrix.
    
    @classmethod
    def frobNorm(cls, m1):
        
        return math.sqrt(cls.frobNorm2(m1))
    
    
    # Normalize a matrix with its Frobenius norm (2-norm).
    
    @classmethod
    def normalize(cls, m1):
        
        return cls.divms(m1, cls.frobNorm(m1))
    
    
    # Produce a new matrix with each element being the absolute of its
    # corresponding element in the original matrix.
    
    @classmethod
    def abs(cls, m1):
        
        r = len(m1)
        c = len(m1[0])
        
        mout = [None] * r
        for i in range(0, r):
            mout[i] = [0.0] * c
            for j in range(0, c):
                mout[i][j] = abs(m1[i][j])
        
        return mout
    
    
    # Return a copy of a matrix.
    
    @classmethod
    def copy(cls, m1):
        
        r = len(m1)
        
        mout = [None] * r
        for i in range(0, r):
            mout[i] = list(m1[i])
        
        return mout
    
    
    # Compute the cross product of two 2-dimensional vectors, represented by
    # two column matrices. The result is a scalar.
    
    @classmethod
    def cross2(cls, m1, m2):
        
        return m1[0][0] * m2[1][0] - m1[1][0] * m2[0][0]
    
    
    # Compute the cross product of two 3-dimensional vectors, represented by
    # two column matrices. The result is a column matrix representing a vector.
    
    @classmethod
    def cross3(cls, m1, m2):
        
        # 'm1' and 'm2' are supposed to be column matrices.
        
        mout = M.column([0.0] * 3)
        mout[0][0] = m1[1][0] * m2[2][0] - m1[2][0] * m2[1][0]
        mout[1][0] = m1[2][0] * m2[0][0] - m1[0][0] * m2[2][0]
        mout[2][0] = m1[0][0] * m2[1][0] - m1[1][0] * m2[0][0]
        
        return mout
    
    
    # Solve a linear system using Gauss-Seidel method. Terminate upon the
    # iteration number is reached or the solution stops changing significantly.
    
    @classmethod
    def gauss_seidel(cls, ma, mb, e, maxIt):
        
        # TODO Handle zeros on diagonal.
        
        r = len(ma)
        c = len(ma[0])
        
        it = 0
        
        mx = M.column([0.0] * r)
        mx0 = M.copy(mx)
        
        while it < maxIt and (it < 1 or M.maxNorm(M.subm(mx, mx0)) > e):
            
            mx0 = M.copy(mx)
            
            for i in range(0, r):
                
                mx[i][0] = mb[i][0]
                for j in range(0, c):
                    if j != i:
                        mx[i][0] -= mx[j][0] * ma[i][j]
                
                mx[i][0] /= ma[i][i]
            
            it += 1
        
        return mx


# Test M class.

def testM():
    
    m1 = [[1, 2, -3], [4, -5, -6]]
    m2 = [[10, 12, 14], [11, 13, 15]]
    print("m1 = " + str(m1))
    print("m2 = " + str(m2))
    print("m1 + m2 = " + str(M.addm(m1, m2)))
    print("m1 - m2 = " + str(M.subm(m1, m2)))
    print("m1^t = " + str(M.transposem(m1)))
    print("m1 * m2^t = " + str(M.multmm(m1, M.transposem(m2))))
    print("m1^t * m2 = " + str(M.multmm(M.transposem(m1), m2)))
    s1 = 0.3
    print("s1 = " + str(s1))
    print("m1 * s1 = " + str(M.multms(m1, s1)))
    l1 = [1, 2, 3, 4]
    print("l1 = " + str(l1))
    print("row(l1) = " + str(M.row(l1)))
    print("column(l1) = " + str(M.column(l1)))
    print("abs(m1) = " + str(M.abs(m1)))
    print("maxNorm(m1) = " + str(M.maxNorm(m1)))
    ma = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]]
    mb = [[-1], [2], [3]]
    print("ma = " + str(ma))
    print("mb = " + str(mb))
    print("gauss_seidel(ma, mb, 1e-3, 100) = ", M.gauss_seidel(ma, mb, 1e-3, 100))
    print("identity_3x3 = " + str(M.diag([1.0] * 3)))
    print("frobNorm2(m1) = " + str(M.frobNorm2(m1)))
    print("frobNorm(m1) = " + str(M.frobNorm(m1)))
    print("normalize(m1) = " + str(M.normalize(m1)))

    m3 = M.column([3, -2, 1])
    m4 = M.column([1, 5, -2])
    print("m3 cross m4 = " + str(M.cross3(m3, m4)))


testM()


# The dependency node class for adaptive remeshing cloth.

class AdaptiveRemeshClothNode(OpenMayaMPx.MPxNode):
    
    # Node name and ID.
    
    kPluginNodeName = "adaptiveRemeshClothNode"
    kPluginNodeId = OpenMaya.MTypeId(0x00033335)
    
    # Node attributes.
    
    # Input mesh as cloth geometry.
    inputMeshAttr = None
    kInputMeshAttrName = "imsh"
    kInputMeshAttrLongName = "inputMesh"
    
    # Output mesh after each simulation step.
    outputMeshAttr = None
    kOutputMeshAttrName = "omsh"
    kOutputMeshAttrLongName = "outputMesh"
    
    # Current time of the simulation.
    currentTimeAttr = None
    kCurrentTimeAttrName = "ctm"
    kCurrentTimeAttrLongName = "currentTime"
    
    # The mesh data of the cloth at the current time step of the simulation.
    currentMesh = None
    
    # Constants for blind data types.
    
    # Blind data IDs. Once for each component type.
    
    vertexBlindDataId = 60
    edgeBlindDataId = 61
    faceBlindDataId = 62
    
    # Default values for various blind data attributes.
    
    bdDefaultVertexMass = 1
    bdDefaultInitialVelocity = [0.0, 0.0, 0.0]
    gravityAcc = [0.0, -9.8, 0.0]
    bdDefaultExternalForce = [0.0, 0.0, 0.0]
    bdDefaultVertexConstraint = False
    bdDefaultStiffness = 1000.0
    bdDefaultMaxStretchRate = 1.05
    bdDefaultMaxCompressRate = 0.95
    bdDefaultFaceSizingField = [[0, 0], [0, 0]]
    
    # Blind data attribute long names.
    
    bdVelocityXLongName = "velocity_x"
    bdVelocityYLongName = "velocity_y"
    bdVelocityZLongName = "velocity_z"
    bdForceXLongName = "force_x"
    bdForceYLongName = "force_y"
    bdForceZLongName = "force_z"
    bdFaceSizingField00LongName = "face_sizing_00"
    bdFaceSizingField01LongName = "face_sizing_01"
    bdFaceSizingField10LongName = "face_sizing_10"
    bdFaceSizingField11LongName = "face_sizing_11"
    bdVertexMassLongName = "mass"
    bdVertexConstraintLongName = "constraint"
    bdEdgeRestLengthLongName = "rest_length"
    bdEdgeRestLengthWUvRatioLongName = "r_wuv_ratio"
    bdStiffnessLongName = "stiffness"
    bdMaxStretchRateLongName = "max_stretch"
    bdMaxCompressRateLongName = "max_compress"
    
    # Blind data attribute short names.
    
    bdVelocityXShortName = "vx"
    bdVelocityYShortName = "vy"
    bdVelocityZShortName = "vz"
    bdForceXShortName = "fx"
    bdForceYShortName = "fy"
    bdForceZShortName = "fz"
    bdFaceSizingField00ShortName = "f_sizing00"
    bdFaceSizingField01ShortName = "f_sizing01"
    bdFaceSizingField10ShortName = "f_sizing10"
    bdFaceSizingField11ShortName = "f_sizing11"
    bdVertexMassShortName = "m"
    bdVertexConstraintShortName = "cnst"
    bdEdgeRestLengthShortName = "r_len"
    bdEdgeRestLengthWUvRatioShortName = "r_wuv"
    bdStiffnessShortName = "stiff"
    bdMaxStretchRateShortName = "m_stretch"
    bdMaxCompressRateShortName = "m_compress"
    
    
    # Constructor.
    
    def __init__(self):
        
        OpenMayaMPx.MPxNode.__init__(self)
        self.currentMesh = OpenMaya.MFnMeshData().create()
        
        # Air resistance factor of the fluid in which the cloth resides.
        self.airResistanceFactor = 5
        
        # Damping factor of the cloth.
        self.dampingFactor = 0.2
        
        # The minimum length of a edge that can be split (in order to prevent
        # too many polygons from generating).
        self.minSplitEdgeLength = 1.2
        
        # The threshold of an edge size so that the edge can be split.
        # Since I did not normalize the sizing fields, the value set here
        # can exceed 1.
        self.minSplitEdgeSize2 = 3.0
        
        # The density of the cloth per unit UV area.
        self.density = 100.0
        
        # Set up dynamic inverse constraints for given vertices.
        self.constraintVerts = [0, 3]
        
        # Time step.
        self.timeStep = 1.0 / 80
        
    
    # Initializer.
    
    @classmethod
    def nodeInitializer(cls):
        
        print(cls.kPluginNodeName + ": Initializing attributes...")
        
        # Set up attributes for input mesh and output mesh.
        
        fnTypedAttr = OpenMaya.MFnTypedAttribute()
        
        cls.inputMeshAttr = fnTypedAttr.create(
            cls.kInputMeshAttrLongName,
            cls.kInputMeshAttrName,
            OpenMaya.MFnData.kMesh
        )
        fnTypedAttr.setWritable(True)
        fnTypedAttr.setStorable(True)
        fnTypedAttr.setHidden(True)
        fnTypedAttr.setArray(False)
        
        cls.outputMeshAttr = fnTypedAttr.create(
            cls.kOutputMeshAttrLongName,
            cls.kOutputMeshAttrName,
            OpenMaya.MFnData.kMesh
        )
        fnTypedAttr.setWritable(False)
        fnTypedAttr.setStorable(True)
        fnTypedAttr.setHidden(True)
        fnTypedAttr.setArray(False)
        
        # Set up the attribute for current time.
        
        fnUnitAttr = OpenMaya.MFnUnitAttribute()
        
        cls.currentTimeAttr = fnUnitAttr.create(
            cls.kCurrentTimeAttrLongName,
            cls.kCurrentTimeAttrName,
            OpenMaya.MFnUnitAttribute.kTime, 0.0
        )
        
        # Add the attributes and their dependencies.
        
        cls.addAttribute(cls.inputMeshAttr)
        cls.addAttribute(cls.outputMeshAttr)
        cls.addAttribute(cls.currentTimeAttr)
        
        cls.attributeAffects(cls.inputMeshAttr, cls.outputMeshAttr)
        cls.attributeAffects(cls.currentTimeAttr, cls.outputMeshAttr)
        
        print(cls.kPluginNodeName + ": Attributes initialized.")
        
    
    # Creator.
    
    @classmethod
    def nodeCreator(cls):
        return OpenMayaMPx.asMPxPtr(cls())
    
    
    # ################ Primitive remeshing opeartions. ################
    
    
    # Primitive remeshing operation: split.
    
    def split(self, fnMesh, edge, newFaces):
        
        util = OpenMaya.MScriptUtil()
        util.createFromInt(0, 0)
        int2Ptr = util.asInt2Ptr()
        
        # Get the end points of the edge to split.
        
        fnMesh.getEdgeVertices(edge, int2Ptr)
        vert0 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 0)
        vert1 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 1)
        
        # Get the faces connected to the given edge.
        
        edgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
        edgeIt.setIndex(edge, util.asIntPtr())
        connectedFaces = OpenMaya.MIntArray()
        edgeIt.getConnectedFaces(connectedFaces)
        
        # Get all vertices of the faces connected to this edge.
        
        connectedPolygonVertices = OpenMaya.MIntArray()
        for j in range(0, connectedFaces.length()):
            curPolygonVertices = OpenMaya.MIntArray()
            fnMesh.getPolygonVertices(connectedFaces[j], curPolygonVertices)
            connectedPolygonVertices = connectedPolygonVertices + curPolygonVertices
        
        # Construct the lists used as parameters for MFnMesh.split().
        
        splitPlacements = OpenMaya.MIntArray()
        splitEdges = OpenMaya.MIntArray()
        splitEdgeFactors = OpenMaya.MFloatArray()
        splitPoints = OpenMaya.MFloatPointArray()
        
        # Compute the edges and edge factors used for 'MFnMesh.split()'. 
        # Since we assume the input mesh is manifold, at most two faces are
        # connected to an edge.
        
        for j in range(0, connectedFaces.length()):
             
            polygonIt = OpenMaya.MItMeshPolygon(self.currentMesh)
            polygonIt.setIndex(connectedFaces[j], util.asIntPtr())
             
            # Get the edges and vertices of a connected face.
              
            polygonEdges = OpenMaya.MIntArray()
            polygonVertices = OpenMaya.MIntArray()
            polygonIt.getVertices(polygonVertices)
            polygonIt.getEdges(polygonEdges)
              
            # The edge and the vertex on which the split is to take place. 
              
            splitEdge = list(set(polygonEdges) - set([edge]))[0]
            splitVert = list(set(polygonVertices) - set([vert0, vert1]))[0]
              
            # Compute the edge factor for this edge based on its direction.
            # We assume that the mesh is already triangulated. 
            
            int2Ptr = util.asInt2Ptr()
            fnMesh.getEdgeVertices(splitEdge, int2Ptr)
            splitVert0 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 0)
            if splitVert0 == splitVert:
                splitFactor = 0.0
            else:
                splitFactor = 1.0
            
            # Add the placement type, the edge and the edge factor to the lists.
            
            splitPlacements.append(OpenMaya.MFnMesh.kOnEdge)
            splitEdges.append(splitEdge)
            splitEdgeFactors.append(splitFactor)
        
        # Insert the edge to split in the middle of the list.
         
        splitPlacements.append(OpenMaya.MFnMesh.kOnEdge)
        splitEdges.insert(edge, 1)
        splitEdgeFactors.insert(0.5, 1)
        
        # Do the split on the two faces.
        
        fnMesh.split(splitPlacements, splitEdges, splitEdgeFactors, splitPoints)
        
        # Get the newly created vertex.
        
        vertIt0 = OpenMaya.MItMeshVertex(self.currentMesh)
        vertIt1 = OpenMaya.MItMeshVertex(self.currentMesh)
        vertIt0.setIndex(vert0, util.asIntPtr())
        vertIt1.setIndex(vert1, util.asIntPtr())
        connectedVerts0 = OpenMaya.MIntArray()
        connectedVerts1 = OpenMaya.MIntArray()
        vertIt0.getConnectedVertices(connectedVerts0)
        vertIt1.getConnectedVertices(connectedVerts1)
        newVert = list(set(connectedVerts0) & set(connectedVerts1) - set(connectedPolygonVertices))[0]
                  
        # Get the faces connected to the newly created vertex.
        
        newVertIt = OpenMaya.MItMeshVertex(self.currentMesh)
        newVertIt.setIndex(newVert, util.asIntPtr())
        newVertIt.getConnectedFaces(newFaces)
        
    
    # Primitive remeshing operation: flip.
    
    def flip(self, fnMesh, edges, newFaces, newEdges):
        
        # Sort the edges to flip in descending order so that the operation on
        # one edge does not disrupt the indices of the others.
        
        edges.sort(reverse = True)
        
        otherVerts0 = []
        otherVerts1 = []
        
        util = OpenMaya.MScriptUtil()
        util.createFromInt(0, 0)
        
        for edge in edges:
            
            edgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
            edgeIt.setIndex(edge, util.asIntPtr())
            
            # Edges on boundaries cannot be flipped.
            if edgeIt.onBoundary():
                continue
            
            # Get the end points of the edge specified.
            
            vertsInt2 = util.asInt2Ptr()
            fnMesh.getEdgeVertices(edge, vertsInt2)
            vert0 = OpenMaya.MScriptUtil.getInt2ArrayItem(vertsInt2, 0, 0)
            vert1 = OpenMaya.MScriptUtil.getInt2ArrayItem(vertsInt2, 0, 1)
            
            # Get the other vertices of the faces connected to the edge to flip.
            
            vertIt0 = OpenMaya.MItMeshVertex(self.currentMesh)
            vertIt1 = OpenMaya.MItMeshVertex(self.currentMesh)
            vertIt0.setIndex(vert0, util.asIntPtr())
            vertIt1.setIndex(vert1, util.asIntPtr())
            connectedVerts0 = OpenMaya.MIntArray()
            connectedVerts1 = OpenMaya.MIntArray()
            vertIt0.getConnectedVertices(connectedVerts0)
            vertIt1.getConnectedVertices(connectedVerts1)
            otherVerts = list(set(connectedVerts0) & set(connectedVerts1) - set([vert0, vert1]))
            
            # Delete the original edge.
            fnMesh.deleteEdge(edge)
            
            # Split the newly generated quad (since we deleted an edge shared by
            # two triangles) along the other diagonal to achieve the effect of
            # flipping the original edge.
            
            splitPlacements = OpenMaya.MIntArray()
            splitEdges = OpenMaya.MIntArray()
            splitEdgeFactors = OpenMaya.MFloatArray()
            splitPoints = OpenMaya.MFloatPointArray()
            
            connectedEdges0 = OpenMaya.MIntArray()
            vertIt0.getConnectedEdges(connectedEdges0)
            connectedEdgesOther = OpenMaya.MIntArray()
            
            for otherVert in otherVerts:
                otherVertIt = OpenMaya.MItMeshVertex(self.currentMesh)
                otherVertIt.setIndex(otherVert, util.asIntPtr())
                otherVertIt.getConnectedEdges(connectedEdgesOther)
                splitEdge = list(set(connectedEdges0) & set(connectedEdgesOther))[0]
                vertsInt2 = util.asInt2Ptr()
                fnMesh.getEdgeVertices(splitEdge, vertsInt2)
                splitEdgeVert0 = OpenMaya.MScriptUtil.getInt2ArrayItem(vertsInt2, 0, 0)
                if splitEdgeVert0 == otherVert:
                    splitFactor = 0.0
                else:
                    splitFactor = 1.0
                
                splitPlacements.append(OpenMaya.MFnMesh.kOnEdge)
                splitEdges.append(splitEdge)
                splitEdgeFactors.append(splitFactor)
            
            fnMesh.split(splitPlacements, splitEdges, splitEdgeFactors, splitPoints)
            
            # Store the end points of the flipped edge.
            
            otherVerts0.append(otherVerts[0])
            otherVerts1.append(otherVerts[1])
        
        # Retrieve the newly generated faces and edges from the end points of
        # the flipped edges stored previously.
        
        for i in range(0, len(otherVerts0)):
        
            otherVertIt0 = OpenMaya.MItMeshVertex(self.currentMesh)
            otherVertIt1 = OpenMaya.MItMeshVertex(self.currentMesh)
            otherVertIt0.setIndex(otherVerts0[i], util.asIntPtr())
            otherVertIt1.setIndex(otherVerts1[i], util.asIntPtr())
            
            connectedFacesOther0 = OpenMaya.MIntArray()
            connectedFacesOther1 = OpenMaya.MIntArray()
            otherVertIt0.getConnectedFaces(connectedFacesOther0)
            otherVertIt1.getConnectedFaces(connectedFacesOther1)
            newSplitFaces = set(connectedFacesOther0) & set(connectedFacesOther1)
            
            # Add the new faces generated by the flip to the list.
            
            for face in newSplitFaces:
                newFaces.append(face)
            
            connectedEdgesOther0 = OpenMaya.MIntArray()
            connectedEdgesOther1 = OpenMaya.MIntArray()
            otherVertIt0.getConnectedEdges(connectedEdgesOther0)
            otherVertIt1.getConnectedEdges(connectedEdgesOther1)
            newSplitEdges = set(connectedEdgesOther0) & set(connectedEdgesOther1)
            
            # Add the new edges generated by the flip to the list.
            
            for edge in newSplitEdges:
                newEdges.append(edge)
        
    
    # Primitive remeshing operation: collapse.
    
    def collapse(self, fnMesh, edges, dirs):
        
        edgesToCollapse = OpenMaya.MIntArray()
        
        utilInt2 = OpenMaya.MScriptUtil()
        utilInt2.createFromInt(0, 0)
        int2Ptr = utilInt2.asInt2Ptr()
        
        utilInt = OpenMaya.MScriptUtil()
        utilInt.createFromInt(0)
        intPtr = utilInt.asIntPtr()
        
        edgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
        
        for i in range(0, len(edges)):
            
            edge = edges[i]
            
            edgeIt.setIndex(edge, intPtr)
            
            d = dirs[i]
            
            # Do nothing if the supplied edge end point0 index is invalid.
            if d != 0 and d != 1:
                continue
            
            # Get the vertices of the edge.
            # 'vert0' is the vertex to merge to when collapsing takes place.
            fnMesh.getEdgeVertices(edge, int2Ptr)
            vert0 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, d)
            vert1 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 1 - d)
            
            # Before collapsing, set 'vert1' to the position of 'vert0' because
            # Maya's 'MFnMesh.collapseEdges()' will move the merged vertices to
            # their midpoint.
            point0 = OpenMaya.MPoint()
            fnMesh.getPoint(vert0, point0)
            fnMesh.setPoint(vert1, point0)
            
            edgesToCollapse.append(edge)
        
        if edgesToCollapse.length() == 0:
            return
        # Collapse the specified edges.
        fnMesh.collapseEdges(edgesToCollapse)
    
    
    # ############ Remeshing operations based on the sizing field. ############
    
    
    # Split edges when their sizes exceed some threshold.
    
    def splitEdges(self, fnMesh, ffields, vfields, esize2s):
        
        # Get all edges.
        
        edges = list(range(0, fnMesh.numEdges()))
        
        # Filter to get splittable edges.
        
        bigEdges = map(lambda pair: pair[0], filter(lambda pair: pair[1] > self.minSplitEdgeSize2, zip(edges, esize2s)))
        
        faceIt = OpenMaya.MItMeshPolygon(self.currentMesh)
        edgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
        vertIt = OpenMaya.MItMeshVertex(self.currentMesh)
        
        utilInt = OpenMaya.MScriptUtil()
        utilInt.createFromInt(0)
        intPtr = utilInt.asIntPtr()
        
        utilDouble = OpenMaya.MScriptUtil()
        utilDouble.createFromDouble(0.0)
        doublePtr = utilDouble.asDoublePtr()
        
        newVertsArray = OpenMaya.MIntArray()
        connectedVertsArray = OpenMaya.MIntArray()
        newEdgesArray = OpenMaya.MIntArray()
        newFacesArray = OpenMaya.MIntArray()
        
        splitFaces = set()
        connectedFacesArray = OpenMaya.MIntArray()
        
        for bigEdge in bigEdges:
            
            # print("#big bigEdge #" + str(bigEdge) + ": " + str(esize2s[bigEdge]))
            
            edgeIt.setIndex(bigEdge, intPtr)
            
            # If the edge is on a face with an already split edge, then skip it.
            edgeIt.getConnectedFaces(connectedFacesArray)
            connectedFaces = set(connectedFacesArray)
            
            if len(connectedFaces & splitFaces) > 0:
                continue
            else:
                splitFaces = splitFaces | connectedFaces
            
            edgeIt.getLength(doublePtr)
            edgeLen = OpenMaya.MScriptUtil.getDouble(doublePtr)
            
            if edgeLen <= self.minSplitEdgeLength:
                continue
            
            # Half the rest length.
            newRestLen = self.getEdgeRestLengths(fnMesh, [bigEdge])[0] * 0.5
            self.setEdgeRestLengths(fnMesh, [bigEdge], [newRestLen])
            
            # Split edges!
            self.split(fnMesh, bigEdge, newFacesArray)
            
            # Flip edges if needed.
            self.flipEdges(fnMesh, newFacesArray)
            
            # Get new vertices and edges from new faces. (Remember to remove duplicates.)

            newVerts = set()
            newEdges = set()
            
            for i in range(0, newFacesArray.length()):
                
                face = newFacesArray[i]
                
                faceIt.setIndex(face, intPtr)
                
                faceIt.getVertices(newVertsArray)
                newVerts = newVerts | set(newVertsArray)
                
                faceIt.getEdges(newEdgesArray)
                newEdges = newEdges | set(newEdgesArray)
            
            newVerts = list(newVerts)
            newEdges = list(newEdges)
            newFaces = list(newFacesArray)
            
            newVerts = filter(lambda vert:
                              not fnMesh.hasBlindDataComponentId(vert,
                                                                 OpenMaya.MFn.kMeshVertComponent,
                                                                 self.vertexBlindDataId),
                              newVerts)
        
            newEdges = filter(lambda edge:
                              not fnMesh.hasBlindDataComponentId(edge,
                                                                 OpenMaya.MFn.kMeshEdgeComponent,
                                                                 self.edgeBlindDataId),
                              newEdges)
        
            newFaces = filter(lambda face:
                              not fnMesh.hasBlindDataComponentId(face,
                                                                 OpenMaya.MFn.kMeshPolygonComponent,
                                                                 self.faceBlindDataId),
                              newFaces)
            
            self.initializeBlindData(fnMesh, newVerts, newEdges, newFaces)
            self.setEdgeRestLengthWUvRatios(fnMesh, newEdges, [self.avgWUv] * len(newEdges))
            self.setEdgeRestLengthWithWUv(fnMesh, newEdges)
            
            for newVert in newVerts:
                
                vertIt.setIndex(newVert, intPtr)
                vertIt.getConnectedVertices(connectedVertsArray)
                connectedVerts = list(connectedVertsArray)
                vs = self.getVelocities(fnMesh, connectedVerts)
                avgV = [0.0, 0.0, 0.0]
                for v in vs:
                    avgV[0] += v[0]
                    avgV[1] += v[1]
                    avgV[1] += v[2]
                avgV[0] /= len(vs)
                avgV[1] /= len(vs)
                avgV[2] /= len(vs)
                self.setVelocities(fnMesh, [newVert], [avgV])
    
    
    # Collapse edges when conditions permit.
    
    def collapseEdges(self, fnMesh):
        
        edgesToCollapse = []
        dirs = []
        
        vertIt0 = OpenMaya.MItMeshVertex(self.currentMesh)
        vertIt1 = OpenMaya.MItMeshVertex(self.currentMesh)
        vertIt2 = OpenMaya.MItMeshVertex(self.currentMesh)
        edgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
        connectedEdgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
        faceIt = OpenMaya.MItMeshPolygon(self.currentMesh)
        
        utilInt = OpenMaya.MScriptUtil()
        utilInt.createFromInt(0)
        intPtr = utilInt.asIntPtr()
        
        utilInt2 = OpenMaya.MScriptUtil()
        utilInt2.createFromInt(0, 0)
        int2Ptr = utilInt2.asInt2Ptr()
        
        utilUv = OpenMaya.MScriptUtil()
        utilUv.createFromDouble(0.0, 0.0)
        float2Ptr = utilUv.asFloat2Ptr()
        
        faceEdgesArray = OpenMaya.MIntArray()
        connectedVertsArray0 = OpenMaya.MIntArray()
        connectedVertsArray1 = OpenMaya.MIntArray()
        connectedEdgesArray = OpenMaya.MIntArray()
        
        while not faceIt.isDone():
            
            # Get all its edges.
            
            faceIt.getEdges(faceEdgesArray)
            faceEdges = list(faceEdgesArray)
            
            # Find a collapsible edge and continue to the next face once one is found.
            for edge in faceEdges:
                
                edgeIt.setIndex(edge, intPtr)
                
                found = False
                
                for d in range(0, 2):
                    
                    fnMesh.getEdgeVertices(edge, int2Ptr)
                    vert0 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, d)
                    vert1 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 1 - d)
                    
                    vertIt0.setIndex(vert0, intPtr)
                    vertIt1.setIndex(vert1, intPtr)
                    
                    # Validity testing: The vertex that is not collapsed to (i.e.
                    # this vertex will be moved) should not be connected to an boundary
                    # edge.
                    
                    vertIt1.getConnectedEdges(connectedEdgesArray)
                    collapsible = True
                    for i in range(0, connectedEdgesArray.length()):
                        connectedEdge = connectedEdgesArray[i]
                        connectedEdgeIt.setIndex(connectedEdge, intPtr)
                        if connectedEdgeIt.onBoundary():
                            collapsible = False
                            break
                        
                    if not collapsible:
                        continue
                    
                    vertIt0.getConnectedVertices(connectedVertsArray0)
                    vertIt1.getConnectedVertices(connectedVertsArray1)
                    
                    oppositeVerts = set(connectedVertsArray1) - set(connectedVertsArray0)
                    
                    collapsible = True
                    
                    vertIt0.getUV(float2Ptr)
                    u0 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 0)
                    v0 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 1)
                    vfield0 = self.computeVertexSizingFields(fnMesh, [vert0])[0]
                    
                    for vert2 in oppositeVerts:
                        
                        vertIt2.setIndex(vert2, intPtr)
                        
                        vertIt2.getUV(float2Ptr)
                        u2 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 0)
                        v2 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 1)
                        vfield2 = self.computeVertexSizingFields(fnMesh, [vert2])[0]
                        
                        # Compute the size of the edge whose end points are
                        # this and the opposite vertices.
                        
                        avgField = M.multms(M.addm(vfield0, vfield2), 0.5)
                        uv0 = M.column([u0, v0])
                        uv2 = M.column([u2, v2])
                        uv02 = M.subm(uv0, uv2)
                        
                        esize2 = M.multmm(M.multmm(M.transposem(uv02), avgField), uv02)
                        
                        if esize2 > self.minSplitEdgeSize2 * 10.0:
                            collapsible = False
                            break
                    
                    if collapsible:
                        edgesToCollapse.append(edge)
                        dirs.append(d)
                        found = True
                        break
                
                if found:
                    break
            
            faceIt.next()
        
        self.collapse(fnMesh, edgesToCollapse, dirs)
        
    
    # Flip edges to preserve the quality of the mesh.
    
    def flipEdges(self, fnMesh, faces):
        
        # Get all edges adjacent to faces.
        # Make sure the edges are independent of each other.
        
        edges = set()
        
        utilInt = OpenMaya.MScriptUtil()
        utilInt.createFromInt(0)
        intPtr = utilInt.asIntPtr()
        
        edgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
        faceIt = OpenMaya.MItMeshPolygon(self.currentMesh)
        
        faceEdgesArray = OpenMaya.MIntArray()
        connectedEdgesArray = OpenMaya.MIntArray()
        
        for i in range(0, faces.length()):
            
            face = faces[i]
            
            faceIt.setIndex(face, intPtr)
            faceIt.getEdges(faceEdgesArray)
            
            for j in range(0, faceEdgesArray.length()):
                
                edge = faceEdgesArray[j]
                
                edgeIt.setIndex(edge, intPtr)
                edgeIt.getConnectedEdges(connectedEdgesArray)
                connectedEdges = set(connectedEdgesArray)
                
                if len(connectedEdges & edges) > 0:
                    continue
                else:
                    edges.add(edge)
                    break
        
        edgesToFlip = filter(lambda edge: self.computeFlipCriteria(fnMesh, edge) < 0, edges)
        
        newFaces = OpenMaya.MIntArray()
        newEdges = OpenMaya.MIntArray()
        self.flip(fnMesh, edgesToFlip, newFaces, newEdges)
        
        self.setEdgeRestLengthWithWUv(fnMesh, list(newEdges), [self.avgWUv] * newEdges.length())
        
    
    # A helper function to compute criteria determining whether an edge should
    # be flipped.
    
    def computeFlipCriteria(self, fnMesh, edge):
        
        utilInt = OpenMaya.MScriptUtil()
        utilInt.createFromInt(0)
        intPtr = utilInt.asIntPtr()
        
        utilInt2 = OpenMaya.MScriptUtil()
        utilInt2.createFromInt(0, 0)
        int2Ptr = utilInt2.asInt2Ptr()
        
        utilUv = OpenMaya.MScriptUtil()
        utilUv.createFromDouble(0.0, 0.0)
        uvPtr = utilUv.asFloat2Ptr()
        
        edgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
        edgeIt.setIndex(edge, intPtr)
        
        if edgeIt.onBoundary():
            return None
        
        fnMesh.getEdgeVertices(edge, int2Ptr)
        vert0 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 0)
        vert1 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 1)
        
        vertIt0 = OpenMaya.MItMeshVertex(self.currentMesh)
        vertIt1 = OpenMaya.MItMeshVertex(self.currentMesh)
        vertIt0.setIndex(vert0, intPtr)
        vertIt1.setIndex(vert1, intPtr)
        connectedVerts0 = OpenMaya.MIntArray()
        connectedVerts1 = OpenMaya.MIntArray()
        vertIt0.getConnectedVertices(connectedVerts0)
        vertIt1.getConnectedVertices(connectedVerts1)
        otherVerts = list(set(connectedVerts0) & set(connectedVerts1) - set([vert0, vert1]))
        
        vert2 = otherVerts[0]
        vert3 = otherVerts[1]
        vertIt2 = OpenMaya.MItMeshVertex(self.currentMesh)
        vertIt3 = OpenMaya.MItMeshVertex(self.currentMesh)
        vertIt2.setIndex(vert2, intPtr)
        vertIt3.setIndex(vert3, intPtr)
        
        # Get UV coordinates of the four vertices.
        # Get the sizing field of the four vertices.
        
        vertIt0.getUV(uvPtr)
        u0 = OpenMaya.MScriptUtil.getFloat2ArrayItem(uvPtr, 0, 0)
        v0 = OpenMaya.MScriptUtil.getFloat2ArrayItem(uvPtr, 0, 1)
        
        vertIt1.getUV(uvPtr)
        u1 = OpenMaya.MScriptUtil.getFloat2ArrayItem(uvPtr, 0, 0)
        v1 = OpenMaya.MScriptUtil.getFloat2ArrayItem(uvPtr, 0, 1)
        
        vertIt2.getUV(uvPtr)
        u2 = OpenMaya.MScriptUtil.getFloat2ArrayItem(uvPtr, 0, 0)
        v2 = OpenMaya.MScriptUtil.getFloat2ArrayItem(uvPtr, 0, 1)
        
        vertIt3.getUV(uvPtr)
        u3 = OpenMaya.MScriptUtil.getFloat2ArrayItem(uvPtr, 0, 0)
        v3 = OpenMaya.MScriptUtil.getFloat2ArrayItem(uvPtr, 0, 1)
        
        m_uv12 = M.column([u2 - u1, v2 - v1])
        m_uv02 = M.column([u2 - u0, v2 - v0])
        m_uv03 = M.column([u3 - u0, v3 - v0])
        m_uv13 = M.column([u3 - u1, v3 - v1])
        
        vfield0 = self.computeVertexSizingFields(fnMesh, [vert0])[0]
        vfield1 = self.computeVertexSizingFields(fnMesh, [vert1])[0]
        vfield2 = self.computeVertexSizingFields(fnMesh, [vert2])[0]
        vfield3 = self.computeVertexSizingFields(fnMesh, [vert3])[0]
        
        m_avgVField = M.divms(M.addm(M.addm(M.addm(vfield0, vfield1), vfield2), vfield3), 4.0)
        
        term1 = M.multms(M.multmm(M.multmm(M.transposem(m_uv03), m_avgVField), m_uv13), M.cross2(m_uv12, m_uv02))
        term2 = M.multms(M.multmm(M.multmm(M.transposem(m_uv12), m_avgVField), m_uv02), M.cross2(m_uv03, m_uv13))
        
        return term1 + term2
        
    
    # ################ Manipulating blind data. ################
    
    
    # Set up blind data templates for all component types.
    
    def initializeBlindDataTypes(self, fnMesh):
        
        print(self.kPluginNodeName + ": Setting up vertex blind data type.")
        
        # Set up per-vertex blind data types.
        
        if not fnMesh.isBlindDataTypeUsed(self.vertexBlindDataId):
            
            longNames = []
            shortNames = []
            formatNames = []
            
            longNames.append(self.bdVelocityXLongName)
            longNames.append(self.bdVelocityYLongName)
            longNames.append(self.bdVelocityZLongName)
            shortNames.append(self.bdVelocityXShortName)
            shortNames.append(self.bdVelocityYShortName)
            shortNames.append(self.bdVelocityZShortName)
            formatNames.append("double")
            formatNames.append("double")
            formatNames.append("double")
            
            longNames.append(self.bdForceXLongName)
            longNames.append(self.bdForceYLongName)
            longNames.append(self.bdForceZLongName)
            shortNames.append(self.bdForceXShortName)
            shortNames.append(self.bdForceYShortName)
            shortNames.append(self.bdForceZShortName)
            formatNames.append("double")
            formatNames.append("double")
            formatNames.append("double")
            
            longNames.append(self.bdVertexMassLongName)
            shortNames.append(self.bdVertexMassShortName)
            formatNames.append("double")
            
            longNames.append(self.bdVertexConstraintLongName)
            shortNames.append(self.bdVertexConstraintShortName)
            formatNames.append("boolean")
            
            fnMesh.createBlindDataType(self.vertexBlindDataId, longNames, shortNames, formatNames)
        
        # Set up per-edge blind data types.
        
        if not fnMesh.isBlindDataTypeUsed(self.edgeBlindDataId):
            
            longNames = []
            shortNames = []
            formatNames = []
            
            longNames.append(self.bdEdgeRestLengthLongName)
            longNames.append(self.bdEdgeRestLengthWUvRatioLongName)
            longNames.append(self.bdStiffnessLongName)
            longNames.append(self.bdMaxStretchRateLongName)
            longNames.append(self.bdMaxCompressRateLongName)
            
            shortNames.append(self.bdEdgeRestLengthShortName)
            shortNames.append(self.bdEdgeRestLengthWUvRatioShortName)
            shortNames.append(self.bdStiffnessShortName)
            shortNames.append(self.bdMaxStretchRateShortName)
            shortNames.append(self.bdMaxCompressRateShortName)
            
            formatNames.append("double")
            formatNames.append("double")
            formatNames.append("double")
            formatNames.append("double")
            formatNames.append("double")
            
            fnMesh.createBlindDataType(self.edgeBlindDataId, longNames, shortNames, formatNames)
        
        # Set up per-face blind data types.
        
        if not fnMesh.isBlindDataTypeUsed(self.faceBlindDataId):
            
            longNames = []
            shortNames = []
            formatNames = []
            longNames.append(self.bdFaceSizingField00LongName)
            longNames.append(self.bdFaceSizingField01LongName)
            longNames.append(self.bdFaceSizingField10LongName)
            longNames.append(self.bdFaceSizingField11LongName)
            shortNames.append(self.bdFaceSizingField00ShortName)
            shortNames.append(self.bdFaceSizingField01ShortName)
            shortNames.append(self.bdFaceSizingField10ShortName)
            shortNames.append(self.bdFaceSizingField11ShortName)
            formatNames.append("double")
            formatNames.append("double")
            formatNames.append("double")
            formatNames.append("double")
            
            fnMesh.createBlindDataType(self.faceBlindDataId, longNames, shortNames, formatNames)
    
    
    # Set vertex masses in blind data.
    
    def setVertexMasses(self, fnMesh, verts = [], ms = []):
        
        if not ms:
            ms = [self.bdDefaultVertexMass] * len(verts)
        elif len(verts) != len(ms):
            return False
        
        for i in range(0, len(verts)):
            fnMesh.setDoubleBlindData(verts[i],
                                      OpenMaya.MFn.kMeshVertComponent,
                                      self.vertexBlindDataId,
                                      self.bdVertexMassLongName,
                                      ms[i])
        
        return True
    
    
    # Set vertex velocities in blind data.
    
    def setVelocities(self, fnMesh, verts = [], vels = []):
        
        if not vels:
            vels = [self.bdDefaultInitialVelocity] * len(verts)
        elif len(verts) != len(vels):
            return False
        
        for i in range(0, len(verts)):
            fnMesh.setDoubleBlindData(verts[i],
                                      OpenMaya.MFn.kMeshVertComponent,
                                      self.vertexBlindDataId,
                                      self.bdVelocityXLongName,
                                      vels[i][0])
            fnMesh.setDoubleBlindData(verts[i],
                                      OpenMaya.MFn.kMeshVertComponent,
                                      self.vertexBlindDataId,
                                      self.bdVelocityYLongName,
                                      vels[i][1])
            fnMesh.setDoubleBlindData(verts[i],
                                      OpenMaya.MFn.kMeshVertComponent,
                                      self.vertexBlindDataId,
                                      self.bdVelocityZLongName,
                                      vels[i][2])
        
        return True
        
    
    # Set external forces imposed on given vertices.
    
    def setExternalForces(self, fnMesh, verts = [], forces = []):
        
        if not forces:
            forces = [self.bdDefaultExternalForce] * len(verts)
        elif len(verts) != len(forces):
            return False
        
        for i in range(0, len(verts)):
            fnMesh.setDoubleBlindData(verts[i],
                                      OpenMaya.MFn.kMeshVertComponent,
                                      self.vertexBlindDataId,
                                      self.bdForceXLongName,
                                      forces[i][0])
            fnMesh.setDoubleBlindData(verts[i],
                                      OpenMaya.MFn.kMeshVertComponent,
                                      self.vertexBlindDataId,
                                      self.bdForceYLongName,
                                      forces[i][1])
            fnMesh.setDoubleBlindData(verts[i],
                                      OpenMaya.MFn.kMeshVertComponent,
                                      self.vertexBlindDataId,
                                      self.bdForceZLongName,
                                      forces[i][2])
        
        return True
    
    
    # Set dynamic inverse constraints on given vertices so that their kinematics
    # do not change with dynamics.
    
    def setVertexConstraints(self, fnMesh, verts = [], constraints = []):
        
        if not constraints:
            constraints = [self.bdDefaultVertexConstraint] * len(verts)
        elif len(verts) != len(constraints):
            return False
        
        for i in range(0, len(verts)):
            fnMesh.setBoolBlindData(verts[i],
                                      OpenMaya.MFn.kMeshVertComponent,
                                      self.vertexBlindDataId,
                                      self.bdVertexConstraintLongName,
                                      constraints[i])
        
        return True
    
    
    # Set the world-UV coordinate length ratios of given edges when they are at
    # their rest lengths. If no ratio is given, current world and UV coordinates
    # of mesh vertices will be used for calculating these ratios.
    
    def setEdgeRestLengthWUvRatios(self, fnMesh, edges = [], wuvs = []):
        
        if not wuvs:
            wuvs = self.computeWUvs(fnMesh, edges)
        elif len(edges) != len(wuvs):
            return False
        
        for i in range(0, len(edges)):
            fnMesh.setDoubleBlindData(edges[i],
                                      OpenMaya.MFn.kMeshEdgeComponent,
                                      self.edgeBlindDataId,
                                      self.bdEdgeRestLengthWUvRatioLongName,
                                      wuvs[i])
        
        return True
    
    
    # Set the rest lengths of given edges by multiplying their lengths in UV
    # coordinates by given world-UV coordinate ratios. If no such ratio is
    # specified, the ratios will be retrieved from the blind data for the
    # calculation.
    
    def setEdgeRestLengthWithWUv(self, fnMesh, edges = [], wuvs = []):
        
        if not wuvs:
            wuvs = self.getEdgeRestLengthWUvRatios(fnMesh, edges)
        elif len(edges) != len(wuvs):
            return False
        
        utilInt2 = OpenMaya.MScriptUtil()
        utilInt2.createFromInt(0, 0)
        int2Ptr = utilInt2.asInt2Ptr()
        
        utilInt = OpenMaya.MScriptUtil()
        utilInt.createFromInt(0, 0)
        intPtr = utilInt.asIntPtr()
        
        utilUv = OpenMaya.MScriptUtil()
        utilUv.createFromDouble(0.0, 0.0)
        float2Ptr = utilUv.asFloat2Ptr()
        
        vertIt = OpenMaya.MItMeshVertex(self.currentMesh)
        edgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
        
        for i in range(0, len(edges)):
            
            edge = edges[i]
            
            edgeIt.setIndex(edge, intPtr)
            
            # Get two end points of the edge.
            fnMesh.getEdgeVertices(edge, int2Ptr)
            vert0 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 0)
            vert1 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 1)
            
            # Get UV coordinates of the two end points.
            
            vertIt.setIndex(vert0, intPtr)
            vertIt.getUV(float2Ptr)
            u0 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 0)
            v0 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 1)
            
            vertIt.setIndex(vert1, intPtr)
            vertIt.getUV(float2Ptr)
            u1 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 0)
            v1 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 1)
            
            # Compute edge length in UV coordinates.
            lenUv = M.frobNorm(M.column([u1 - u0, v1 - v0]))
            
            # Multiply the length in UV-coordinates by its corresponding
            # world-UV coordinate ratio.
            lenW = lenUv * wuvs[i]
            
            fnMesh.setDoubleBlindData(edges[i],
                                      OpenMaya.MFn.kMeshEdgeComponent,
                                      self.edgeBlindDataId,
                                      self.bdEdgeRestLengthLongName,
                                      lenW)
        
        return True
    
    
    # Set the rest lengths of given edges. If no length is specified, their
    # current lengths will be used.
    
    def setEdgeRestLengths(self, fnMesh, edges = [], lengths = []):
        
        if not lengths:
            
            lengths = [0.0] * len(edges)
            
            edgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
            utilInt = OpenMaya.MScriptUtil()
            utilInt.createFromInt(0)
            intPtr = utilInt.asIntPtr()
            utilDouble = OpenMaya.MScriptUtil()
            utilDouble.createFromDouble(0.0)
            doublePtr = utilDouble.asDoublePtr()
            
            for i in range(0, len(edges)):
                
                edge = edges[i]
                edgeIt.setIndex(edge, intPtr)
                edgeIt.getLength(doublePtr)
                lengths[i] = OpenMaya.MScriptUtil.getDouble(doublePtr)
            
        elif len(edges) != len(lengths):
            return False
        
        for i in range(0, len(edges)):
            
            fnMesh.setDoubleBlindData(edges[i],
                                      OpenMaya.MFn.kMeshEdgeComponent,
                                      self.edgeBlindDataId,
                                      self.bdEdgeRestLengthLongName,
                                      lengths[i])
        
        return True
    
    
    # Set the stiffness rates of given edges.
    
    def setStiffnesses(self, fnMesh, edges = [], stiffnesses = []):
        
        if not stiffnesses:
            stiffnesses = [self.bdDefaultStiffness] * len(edges)
        elif len(edges) != len(stiffnesses):
            return False
        
        for i in range(0, len(edges)):
            fnMesh.setDoubleBlindData(edges[i],
                                      OpenMaya.MFn.kMeshEdgeComponent,
                                      self.edgeBlindDataId,
                                      self.bdStiffnessLongName,
                                      stiffnesses[i])
        
        return True
    
    
    # Set the maximum compression rates of given edges. If the edge is
    # compressed to shorter than its rest length times this rate, dynamic
    # inverse constraints will take effect and pull it back.
    
    def setMaxCompressRates(self, fnMesh, edges = [], maxCompresses = []):
        
        if not maxCompresses:
            maxCompresses = [self.bdDefaultMaxCompressRate] * len(edges)
        elif len(edges) != len(maxCompresses):
            return False
        
        for i in range(0, len(edges)):
            fnMesh.setDoubleBlindData(edges[i],
                                      OpenMaya.MFn.kMeshEdgeComponent,
                                      self.edgeBlindDataId,
                                      self.bdMaxCompressRateLongName,
                                      maxCompresses[i])
        
        return True
    
    
    # Set the maximum stretching rates of given edges. If the edge is stretched
    # to longer than its rest length times this rate, dynamic inverse
    # constraints will take effect and pull it back.
    
    def setMaxStretchRates(self, fnMesh, edges = [], maxStretches = []):
        
        if not maxStretches:
            maxStretches = [self.bdDefaultMaxStretchRate] * len(edges)
        elif len(edges) != len(maxStretches):
            return False
        
        for i in range(0, len(edges)):
            fnMesh.setDoubleBlindData(edges[i],
                                      OpenMaya.MFn.kMeshEdgeComponent,
                                      self.edgeBlindDataId,
                                      self.bdMaxStretchRateLongName,
                                      maxStretches[i])
        
        return True
    
    
    # Set the sizing fields of given faces.
    
    def setFaceSizingFields(self, fnMesh, faces = [], fields = []):
        
        if not fields:
            fields = [self.bdDefaultFaceSizingField] * len(faces)
        elif len(faces) != len(fields):
            return False
        
        for i in range(0, len(faces)):
            
            fnMesh.setDoubleBlindData(faces[i],
                                      OpenMaya.MFn.kMeshPolygonComponent,
                                      self.faceBlindDataId,
                                      self.bdFaceSizingField00LongName,
                                      fields[i][0][0])
            fnMesh.setDoubleBlindData(faces[i],
                                      OpenMaya.MFn.kMeshPolygonComponent,
                                      self.faceBlindDataId,
                                      self.bdFaceSizingField01LongName,
                                      fields[i][0][1])
            fnMesh.setDoubleBlindData(faces[i],
                                      OpenMaya.MFn.kMeshPolygonComponent,
                                      self.faceBlindDataId,
                                      self.bdFaceSizingField10LongName,
                                      fields[i][1][0])
            fnMesh.setDoubleBlindData(faces[i],
                                      OpenMaya.MFn.kMeshPolygonComponent,
                                      self.faceBlindDataId,
                                      self.bdFaceSizingField11LongName,
                                      fields[i][1][1])
        
        return True
    
    
    # Get the masses of given vertices.
    
    def getVertexMasses(self, fnMesh, verts = []):
        
        vertsArray = OpenMaya.MIntArray()
        massesArray = OpenMaya.MDoubleArray()
        
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshVertComponent,
                                  self.vertexBlindDataId,
                                  self.bdVertexMassLongName,
                                  vertsArray,
                                  massesArray
                                  )
        
        masses = [0.0] * fnMesh.numVertices()
        for i in range(0, massesArray.length()):
            masses[vertsArray[i]] = massesArray[i]
        
        return [masses[i] for i in verts]
    
    
    # Compute the area represented by each vertex in the give vertex list, which
    # is one third of the area sum of the faces connected to this vertex.
    # The areas are considered in the UV space.
    
    def computeVertexArea(self, fnMesh, verts = []):
        
        vertIt = OpenMaya.MItMeshVertex(self.currentMesh)
        faceIt = OpenMaya.MItMeshPolygon(self.currentMesh)
        
        utilInt = OpenMaya.MScriptUtil()
        utilInt.createFromInt(0)
        intPtr = utilInt.asIntPtr()
        
        utilDouble = OpenMaya.MScriptUtil()
        utilDouble.createFromDouble(0.0)
        doublePtr = utilDouble.asDoublePtr()
        
        connectedFacesArray = OpenMaya.MIntArray()
        
        areas = [0.0] * len(verts)
        
        for i in range(0, len(verts)):
            
            vert = verts[i]
            vertIt.setIndex(vert, intPtr)
            
            vertIt.getConnectedFaces(connectedFacesArray)
            
            areaSum = 0.0
            for j in range(0, connectedFacesArray.length()):
                
                face = connectedFacesArray[j]
                faceIt.setIndex(face, intPtr)
                
                faceIt.getUVArea(doublePtr)
                area = OpenMaya.MScriptUtil.getDouble(doublePtr)
                areaSum += area
            
            areas[i] = areaSum / 3.0
        
        return areas
        
    
    # Compute the masses of given vertices by multiplying the cloth density by
    # the areas represented by the vertices.
    
    def computeVertexMasses(self, fnMesh, verts = []):
        
        return map(lambda area: area * self.density, self.computeVertexArea(fnMesh, verts))
    
    
    # Get the velocities of given vertices.
    
    def getVelocities(self, fnMesh, verts = []):
        
        vertsArray = OpenMaya.MIntArray()
        vxsArray = OpenMaya.MDoubleArray()
        vysArray = OpenMaya.MDoubleArray()
        vzsArray = OpenMaya.MDoubleArray()
        
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshVertComponent,
                                  self.vertexBlindDataId,
                                  self.bdVelocityXLongName,
                                  vertsArray,
                                  vxsArray
                                  )
        vxs = [0.0] * fnMesh.numVertices()
        for i in range(0, vertsArray.length()):
            vxs[vertsArray[i]] = vxsArray[i]
            
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshVertComponent,
                                  self.vertexBlindDataId,
                                  self.bdVelocityYLongName,
                                  vertsArray,
                                  vysArray
                                  )
        vys = [0.0] * fnMesh.numVertices()
        for i in range(0, vertsArray.length()):
            vys[vertsArray[i]] = vysArray[i]
        
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshVertComponent,
                                  self.vertexBlindDataId,
                                  self.bdVelocityZLongName,
                                  vertsArray,
                                  vzsArray
                                  )
        vzs = [0.0] * fnMesh.numVertices()
        for i in range(0, vertsArray.length()):
            vzs[vertsArray[i]] = vzsArray[i]
        
        return [[vxs[i], vys[i], vzs[i]] for i in verts]
    
    
    # Get the external forces imposed on given vertices.
    
    def getExternalForces(self, fnMesh, verts = []):
        
        vertsArray = OpenMaya.MIntArray(len(verts), 0)
        fxsArray = OpenMaya.MDoubleArray(len(verts), 0)
        fysArray = OpenMaya.MDoubleArray(len(verts), 0)
        fzsArray = OpenMaya.MDoubleArray(len(verts), 0)
        
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshVertComponent,
                                  self.vertexBlindDataId,
                                  self.bdForceXLongName,
                                  vertsArray,
                                  fxsArray
                                  )
        fxs = [0.0] * fnMesh.numVertices()
        for i in range(0, vertsArray.length()):
            fxs[vertsArray[i]] = fxsArray[i]
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshVertComponent,
                                  self.vertexBlindDataId,
                                  self.bdForceYLongName,
                                  vertsArray,
                                  fysArray
                                  )
        fys = [0.0] * fnMesh.numVertices()
        for i in range(0, vertsArray.length()):
            fys[vertsArray[i]] = fysArray[i]
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshVertComponent,
                                  self.vertexBlindDataId,
                                  self.bdForceZLongName,
                                  vertsArray,
                                  fzsArray
                                  )
        fzs = [0.0] * fnMesh.numVertices()
        for i in range(0, vertsArray.length()):
            fzs[vertsArray[i]] = fzsArray[i]
                
        return [[fxs[i], fys[i], fzs[i]] for i in verts]
    
    
    # Get the dynamic inverse constraints of given vertices.
    
    def getVertexConstraints(self, fnMesh, verts = []):
        
        vertsArray = OpenMaya.MIntArray()
        constraintsArray = OpenMaya.MIntArray(0)
        fnMesh.getBoolBlindData(OpenMaya.MFn.kMeshVertComponent,
                                self.vertexBlindDataId,
                                self.bdVertexConstraintLongName,
                                vertsArray,
                                constraintsArray
                                )
        
        constraints = [False] * fnMesh.numVertices()
        for i in range(0, vertsArray.length()):
            constraints[vertsArray[i]] = constraintsArray[i]
        
        return map(lambda x: x != 0, [constraints[i] for i in verts])
    
    
    # Get the rest lengths of given edges.
    
    def getEdgeRestLengths(self, fnMesh, edges = []):
        
        edgesArray = OpenMaya.MIntArray()
        lengthsArray = OpenMaya.MDoubleArray()
        
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshEdgeComponent,
                                  self.edgeBlindDataId,
                                  self.bdEdgeRestLengthLongName,
                                  edgesArray,
                                  lengthsArray
                                  )
        
        lengths = [0.0] * fnMesh.numEdges()
        for i in range(0, edgesArray.length()):
            lengths[edgesArray[i]] = lengthsArray[i]
        
        return [lengths[i] for i in edges]
    
    
    # Get the world-UV coordinate ratios of the rest lengths of given edges.
    
    def getEdgeRestLengthWUvRatios(self, fnMesh, edges = []):
        
        edgesArray = OpenMaya.MIntArray()
        wuvsArray = OpenMaya.MDoubleArray()
        
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshEdgeComponent,
                                  self.edgeBlindDataId,
                                  self.bdEdgeRestLengthWUvRatioLongName,
                                  edgesArray,
                                  wuvsArray
                                  )
        
        wuvs = [0.0] * fnMesh.numEdges()
        for i in range(0, edgesArray.length()):
            wuvs[edgesArray[i]] = wuvsArray[i]
        
        return [wuvs[i] for i in edges]
    
    
    # Get the rest lengths of given edges by multiplying their lengths in UV
    # coordinates by their rest-length world-UV coordinate ratio.
    
    def getEdgeRestLengthsWithWUv(self, fnMesh, edges = []):
        
        utilInt2 = OpenMaya.MScriptUtil()
        utilInt2.createFromInt(0, 0)
        int2Ptr = utilInt2.asInt2Ptr()
        
        utilInt = OpenMaya.MScriptUtil()
        utilInt.createFromInt(0, 0)
        intPtr = utilInt.asIntPtr()
        
        utilUv = OpenMaya.MScriptUtil()
        utilUv.createFromDouble(0.0, 0.0)
        float2Ptr = utilUv.asFloat2Ptr()
        
        vertIt = OpenMaya.MItMeshVertex(self.currentMesh)
        edgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
        
        wuvs = self.getEdgeRestLengthWUvRatios(fnMesh, edges)
        lengths = [0.0] * len(edges)
        
        for i in range(0, len(edges)):
            
            edge = edges[i]
            
            edgeIt.setIndex(edge, intPtr)
            
            # Get two end points of the edge.
            fnMesh.getEdgeVertices(edge, int2Ptr)
            vert0 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 0)
            vert1 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 1)
            
            # Get UV coordinates of the two end points.
            
            vertIt.setIndex(vert0, intPtr)
            vertIt.getUV(float2Ptr)
            u0 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 0)
            v0 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 1)
            
            vertIt.setIndex(vert1, intPtr)
            vertIt.getUV(float2Ptr)
            u1 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 0)
            v1 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 1)
            
            # Compute edge length in UV coordinates.
            lenUv = M.frobNorm(M.column([u1 - u0, v1 - v0]))
            
            # Compute edge length in world coordinates.
            lengths[i] = lenUv * wuvs[i]
            
        return lengths
    
    
    # Get the stiffness rates of given edges.
    
    def getStiffnesses(self, fnMesh, edges = []):
        
        edgesArray = OpenMaya.MIntArray()
        stiffnessesArray = OpenMaya.MDoubleArray()
        
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshEdgeComponent,
                                  self.edgeBlindDataId,
                                  self.bdStiffnessLongName,
                                  edgesArray,
                                  stiffnessesArray
                                  )
        
        stiffnesses = [0.0] * fnMesh.numEdges()
        for i in range(0, edgesArray.length()):
            stiffnesses[edgesArray[i]] = stiffnessesArray[i]
        
        return [stiffnesses[i] for i in edges]
    
    
    # Get the maximum compression rates of given edges.
    
    def getMaxCompressRates(self, fnMesh, edges = []):
        
        edgesArray = OpenMaya.MIntArray()
        maxCompressesArray = OpenMaya.MDoubleArray()
        
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshEdgeComponent,
                                  self.edgeBlindDataId,
                                  self.bdMaxCompressRateLongName,
                                  edgesArray,
                                  maxCompressesArray
                                  )
        
        maxCompresses = [0.0] * fnMesh.numEdges()
        for i in range(0, edgesArray.length()):
            maxCompresses[edgesArray[i]] = maxCompressesArray[i]
        
        return [maxCompresses[i] for i in edges]
    
    
    # Get the maximum stretching rates of given edges.
    
    def getMaxStretchRates(self, fnMesh, edges = []):
        
        edgesArray = OpenMaya.MIntArray()
        maxStretchesArray = OpenMaya.MDoubleArray()
        
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshEdgeComponent,
                                  self.edgeBlindDataId,
                                  self.bdMaxStretchRateLongName,
                                  edgesArray,
                                  maxStretchesArray
                                  )
        
        maxStretches = [0.0] * fnMesh.numEdges()
        for i in range(0, edgesArray.length()):
            maxStretches[edgesArray[i]] = maxStretchesArray[i]
        
        return [maxStretches[i] for i in edges]
    
    
    # Get the sizing fields of given faces.
    
    def getFaceSizingFields(self, fnMesh, faces = []):
        
        facesArray = OpenMaya.MIntArray()
        field00sArray = OpenMaya.MDoubleArray()
        field01sArray = OpenMaya.MDoubleArray()
        field10sArray = OpenMaya.MDoubleArray()
        field11sArray = OpenMaya.MDoubleArray()
        
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshPolygonComponent,
                                  self.faceBlindDataId,
                                  self.bdFaceSizingField00LongName,
                                  facesArray,
                                  field00sArray
                                  )
        field00s = [0.0] * fnMesh.numPolygons()
        for i in range(0, facesArray.length()):
            field00s[facesArray[i]] = field00sArray[i]
            
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshPolygonComponent,
                                  self.faceBlindDataId,
                                  self.bdFaceSizingField01LongName,
                                  facesArray,
                                  field01sArray
                                  )
        field01s = [0.0] * fnMesh.numPolygons()
        for i in range(0, facesArray.length()):
            field01s[facesArray[i]] = field01sArray[i]
        
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshPolygonComponent,
                                  self.faceBlindDataId,
                                  self.bdFaceSizingField10LongName,
                                  facesArray,
                                  field10sArray
                                  )
        field10s = [0.0] * fnMesh.numPolygons()
        for i in range(0, facesArray.length()):
            field10s[facesArray[i]] = field10sArray[i]
        
        fnMesh.getDoubleBlindData(OpenMaya.MFn.kMeshPolygonComponent,
                                  self.faceBlindDataId,
                                  self.bdFaceSizingField11LongName,
                                  facesArray,
                                  field11sArray
                                  )
        field11s = [0.0] * fnMesh.numPolygons()
        for i in range(0, facesArray.length()):
            field11s[facesArray[i]] = field10sArray[i]
        
        return [[[field00s[i], field01s[i]], [field10s[i], field11s[i]]] for i in faces]
    
    
    # Initialize the blind data of all specified mesh components (i.e. vertices,
    # edges and faces) to default values. 
    
    def initializeBlindData(self, fnMesh, verts = [], edges = [], faces = []):
                
        self.setVertexMasses(fnMesh, verts)
        self.setVelocities(fnMesh, verts)
        self.setExternalForces(fnMesh, verts)
        self.setVertexConstraints(fnMesh, verts)
        
        self.setEdgeRestLengths(fnMesh, edges)
        self.setEdgeRestLengthWUvRatios(fnMesh, edges)
        self.setStiffnesses(fnMesh, edges)
        self.setMaxCompressRates(fnMesh, edges)
        self.setMaxStretchRates(fnMesh, edges)
        
        self.setFaceSizingFields(fnMesh, faces)
    
    
    # ######## Basic cloth simulation cycle implementation. ########
    
    
    # Time step integration.
    
    def step(self, fnMesh, dt):
        
        # Get attributes of mesh components used for simulation.
        
        # Vertex masses.
        ms = self.computeVertexMasses(fnMesh,
                                  list(range(0, fnMesh.numVertices())))
        # Vertex velocities.
        vs = self.getVelocities(fnMesh,
                                list(range(0, fnMesh.numVertices())))
        # Vertex external forces.
        fs = self.getExternalForces(fnMesh,
                                    list(range(0, fnMesh.numVertices())))
        # Vertex dynamic inverse constraints.
        cs = self.getVertexConstraints(fnMesh,
                                       list(range(0, fnMesh.numVertices())))
        # Edge rest lengths computed with their rest-length world-UV coordinate
        # ratio.
        l0s = self.getEdgeRestLengthsWithWUv(fnMesh,
                                             list(range(0, fnMesh.numEdges())))
        # Edge stiffness rates.
        stiffs = self.getStiffnesses(fnMesh,
                                     list(range(0, fnMesh.numEdges())))
        # Edge maximum compression rates.
        climits = self.getMaxCompressRates(fnMesh,
                                           list(range(0, fnMesh.numEdges())))
        # Edge maximum stretching rates.
        slimits = self.getMaxStretchRates(fnMesh,
                                          list(range(0, fnMesh.numEdges())))
        
        areas = self.computeVertexArea(fnMesh,
                                    list(range(0, fnMesh.numVertices())))
        
        # Get the positions of all vertices.
        xsArray = OpenMaya.MPointArray()
        fnMesh.getPoints(xsArray)
        
        # Compute force related terms.
        m_f0s, m_df_dxs, m_df_dvs = self.computeForceTerms(dt, fnMesh, xsArray, ms, vs, fs, cs, l0s, stiffs, climits, slimits, areas)
        
        # ######## Forward Euler integration. Not in use in the final version. ########
#         
#         # Update forces.
#         fs = map(lambda m_f0: [m_f0[0][0], m_f0[1][0], m_f0[2][0]] if m_f0 != None else [0.0, 0.0, 0.0], m_f0s)
#          
#         # Update velocities.
#         for vert in range(0, len(vs)): 
#             vs[vert][0] += fs[vert][0] / ms[vert] * dt
#             vs[vert][1] += fs[vert][1] / ms[vert] * dt
#             vs[vert][2] += fs[vert][2] / ms[vert] * dt
#         self.setVelocities(fnMesh, list(range(0, fnMesh.numVertices())), vs)
        
        # ######## Backward Euler integration. ########
        
        m_v0s = map(lambda v: M.column(v), vs)
        m_m_invs = map(lambda m: M.diag([1.0 / m] * 3), ms)
 
        for vert in range(0, fnMesh.numVertices()):
            
            if cs[vert]:
                continue
              
            m_m_inv = m_m_invs[vert]
            m_df_dx = m_df_dxs[vert]
            m_df_dv = m_df_dvs[vert]
            m_f0 = m_f0s[vert]
            m_v0 = m_v0s[vert]
              
            m_a = M.subm(M.subm(M.diag([1.0] * 3), M.multms(M.multmm(m_m_inv, m_df_dx), dt * dt)), M.multms(M.multmm(m_m_inv, m_df_dv), dt))
            m_b = M.multms(M.multmm(m_m_inv, M.addm(m_f0, M.multms(M.multmm(m_df_dx, m_v0), dt))), dt)
              
            m_dv = M.gauss_seidel(m_a, m_b, 1e-6, 20)
            
            vs[vert][0] += m_dv[0][0]
            vs[vert][1] += m_dv[1][0]
            vs[vert][2] += m_dv[2][0]
            
        self.setVelocities(fnMesh, list(range(0, fnMesh.numVertices())), vs)
        
        # Update positions.
        self.updatePositions(dt, fnMesh, xsArray, ms, vs, fs, cs, l0s, stiffs, climits, slimits)
        fnMesh.setPoints(xsArray)
    
    
    # Compute force related terms, including force, partial derivatives of
    # forces to positions and velocities, etc.
    
    def computeForceTerms(self, dt, fnMesh, xsArray, ms, vs, fs, cs, l0s, stiffs, climits, slimits, areas):
        
        vertIt = OpenMaya.MItMeshVertex(self.currentMesh)
        
        connectedEdges = OpenMaya.MIntArray()
        
        util = OpenMaya.MScriptUtil()
        util.createFromInt(0)
        intPtr= util.asIntPtr()
        
        # Forces.
        m_f0s = [None] * len(fs)
        
        # Partial derivatives of forces to positions.
        m_df_dxs = [None] * len(fs)
        m_df_dvs = [None] * len(fs)
        
        while not vertIt.isDone():
            
            vert = vertIt.index()
            
            if not cs[vert]:
                
                # Initialize the force vector to the external force by
                # copying.
                f = list(fs[vert])
                
                # Vertex mass.
                m = ms[vert]
                
                # Jacobians of force with respect to position and velocity (df/dx and df/dv).
                df_dx = M.zero(3, 3)
                df_dv = M.zero(3, 3)
                
                # Gravity force, which is not dependent on position.
                f[0] += self.gravityAcc[0] * m
                f[1] += self.gravityAcc[1] * m
                f[2] += self.gravityAcc[2] * m
                
                # Get velocity and velocity direction (normalized velocity
                # vector).
                m_v = M.column(vs[vert])
                vnorm = M.frobNorm(m_v)
                
                # Compute the air resistance force, which is quadratic to the vertex
                # velocity.
                f[0] += -self.airResistanceFactor * areas[vert] * vnorm * m_v[0][0]
                f[1] += -self.airResistanceFactor * areas[vert] * vnorm * m_v[1][0]
                f[2] += -self.airResistanceFactor * areas[vert] * vnorm * m_v[2][0]
                
                # Update df/dv if necessary.
                
                if vnorm != 0:
                
                    df_dv[0][0] = -self.airResistanceFactor * areas[vert] * (0.5 / vnorm * 2 * m_v[0][0] * m_v[0][0] + vnorm)
                    df_dv[0][1] = -self.airResistanceFactor * areas[vert] * (0.5 / vnorm * 2 * m_v[0][0] * m_v[0][0])
                    df_dv[0][2] = -self.airResistanceFactor * areas[vert] * (0.5 / vnorm * 2 * m_v[0][0] * m_v[0][0])
                    
                    df_dv[1][0] = -self.airResistanceFactor * areas[vert] * (0.5 / vnorm * 2 * m_v[1][0] * m_v[1][0])
                    df_dv[1][1] = -self.airResistanceFactor * areas[vert] * (0.5 / vnorm * 2 * m_v[1][0] * m_v[1][0] + vnorm)
                    df_dv[1][2] = -self.airResistanceFactor * areas[vert] * (0.5 / vnorm * 2 * m_v[1][0] * m_v[1][0])
                    
                    df_dv[2][0] = -self.airResistanceFactor * areas[vert] * (0.5 / vnorm * 2 * m_v[2][0] * m_v[2][0])
                    df_dv[2][1] = -self.airResistanceFactor * areas[vert] * (0.5 / vnorm * 2 * m_v[2][0] * m_v[2][0])
                    df_dv[2][2] = -self.airResistanceFactor * areas[vert] * (0.5 / vnorm * 2 * m_v[2][0] * m_v[2][0] + vnorm)
                
                # Get edges connected to this vertex.
                vertIt.getConnectedEdges(connectedEdges)
                
                for i in range(0, connectedEdges.length()):
                    
                    connectedEdge = connectedEdges[i]
                    
                    # Get the vertex on the other side of the connect edge.
                    vertIt.getOppositeVertex(intPtr, connectedEdge)
                    connectedVert = OpenMaya.MScriptUtil.getInt(intPtr)
                    
                    # Compute current length of the connected edge.
                    x = xsArray[vert]
                    x0 = xsArray[connectedVert]
                    offset = x0 - x
                    length = x.distanceTo(x0)
                    direction = offset / length
                    
                    # Get the rest length of the connected edge.
                    l0 = l0s[connectedEdge]
                    
                    # Get the stiffness rate.
                    stiff = stiffs[connectedEdge]
                    
                    # Compute stretch/compress force contributed by this edge.
                    springForce = direction * (length - l0) * stiff
                    f[0] += springForce.x
                    f[1] += springForce.y
                    f[2] += springForce.z
                    
                    # Update df/dx if necessary.
                    
                    if length > 0:
                    
                        df_dx[0][0] += stiff * ((l0 / length - 1) + (x.x - x0.x) * (-0.5 * l0 / (length * length * length) * 2 * (x.x - x0.x)))
                        df_dx[0][1] += stiff * ((x.x - x0.x) * (-0.5 * l0 / (length * length * length) * 2 * (x.y - x0.y)))
                        df_dx[0][2] += stiff * ((x.x - x0.x) * (-0.5 * l0 / (length * length * length) * 2 * (x.z - x0.z)))
                        
                        df_dx[1][0] += stiff * ((x.y - x0.y) * (-0.5 * l0 / (length * length * length) * 2 * (x.x - x0.x)))
                        df_dx[1][1] += stiff * ((l0 / length - 1) + (x.y - x0.y) * (-0.5 * l0 / (length * length * length) * 2 * (x.y - x0.y)))
                        df_dx[1][2] += stiff * ((x.y - x0.y) * (-0.5 * l0 / (length * length * length) * 2 * (x.z - x0.z)))
                        
                        df_dx[2][0] += stiff * ((x.z - x0.z) * (-0.5 * l0 / (length * length * length) * 2 * (x.x - x0.x)))
                        df_dx[2][1] += stiff * ((x.z - x0.z) * (-0.5 * l0 / (length * length * length) * 2 * (x.y - x0.y)))
                        df_dx[2][2] += stiff * ((l0 / length - 1) + (x.z - x0.z) * (-0.5 * l0 / (length * length * length) * 2 * (x.z - x0.z)))
                    
                    # Update df/dv if necessary.
                
                # Store the force and partial derivatives as matrices.
                m_f0s[vert] = M.column(f)
                m_df_dxs[vert] = df_dx
                m_df_dvs[vert] = df_dv
            
            vertIt.next()
        
        return (m_f0s, m_df_dxs, m_df_dvs)
    
    
    # Update positions with updated velocities and apply dynamic inverse
    # constraints.
    
    def updatePositions(self, dt, fnMesh, xsArray, ms, vs, fs, cs, l0s, stiffs, climits, slimits):
        
        vertIt = OpenMaya.MItMeshVertex(self.currentMesh)
        edgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
        
        utilInt = OpenMaya.MScriptUtil()
        utilInt.createFromInt(0)
        intPtr = utilInt.asIntPtr()
        
        connectedEdges = OpenMaya.MIntArray()
        
        while not vertIt.isDone():
            
            vert = vertIt.index()
            
            # Position.
            x = xsArray[vert]
            # Updated velocity.
            v = vs[vert]
            
            # Step the position with update velocity.
            x.x += v[0] * dt
            x.y += v[1] * dt
            x.z += v[2] * dt
            
            # Get connected edges.
            vertIt.getConnectedEdges(connectedEdges)
            
            for i in range(0, connectedEdges.length()):
                
                edge = connectedEdges[i]
                edgeIt.setIndex(edge, intPtr)
                
                # Get the position of two end points of the connected edge.
                vert0 = vert
                vertIt.getOppositeVertex(intPtr, edge)
                vert1 = OpenMaya.MScriptUtil.getInt(intPtr)
                x0 = xsArray[vert0]
                x1 = xsArray[vert1]
                
                # Get the offset, length and direction of the connected edge.
                offset = x1 - x0
                length = x1.distanceTo(x0)
                direction = offset / length
                 
                # Get rest length.
                l0 = l0s[edge]
                
                # Maximum compression rate.
                climit = climits[edge]
                # Maximum stretching rate.
                slimit = slimits[edge]
                
                # Maximum length due to the constraint.
                lmax = slimit * l0
                # Minimum length due to the constraint.
                lmin = climit * l0
                
                # Compute the amount to offset the end point(s) of the edge
                # if it is longer or shorter than the maximum or minimum
                # length allowed by the constraint. The amount will be set to
                # zero if the length is within the limits.
                
                if length > lmax:
                    coffset = length - lmax
                elif length < lmin:
                    coffset = length - lmin
                else:
                    coffset = 0.0
                
                # Apply the constraint only on unconstrained end points. If both
                # end points are unconstrained, the offset amount will be halved
                # and applied to both of them along opposite direction.
                
                if not cs[vert0]:
                    
                    if not cs[vert1]:
                        x0.x += 0.5 * coffset * direction.x
                        x0.y += 0.5 * coffset * direction.y
                        x0.z += 0.5 * coffset * direction.z
                    else:
                        x0.x += coffset * direction.x
                        x0.y += coffset * direction.y
                        x0.z += coffset * direction.z
            
                if not cs[vert1]:
                    
                    if not cs[vert0]:
                        x1.x -= 0.5 * coffset * direction.x
                        x1.y -= 0.5 * coffset * direction.y
                        x1.z -= 0.5 * coffset * direction.z
                    else:
                        x1.x -= coffset * direction.x
                        x1.y -= coffset * direction.y
                        x1.z -= coffset * direction.z
            
            vertIt.next()
    
    
    # ################ Sizing field computation. ################
    
    
    # Compute the sizing fields of given faces.
    
    def computeFaceSizingFields(self, fnMesh, faces = []):
        
        ffields = [None] * len(faces)
        
        uArray = OpenMaya.MFloatArray()
        vArray = OpenMaya.MFloatArray()
        fnMesh.getUVs(uArray, vArray)
        
        faceIt = OpenMaya.MItMeshPolygon(self.currentMesh)
        
        utilUv = OpenMaya.MScriptUtil()
        utilUv.createFromDouble(0.0, 0.0)
        float2Ptr = utilUv.asFloat2Ptr()
        
        utilIdx = OpenMaya.MScriptUtil()
        utilIdx.createFromInt(0)
        intPtr = utilIdx.asIntPtr()
        
        faceVertsArray = OpenMaya.MIntArray()
        
        for i in range(0, len(faces)):
            
            face = faces[i]
            
            faceIt.setIndex(face, intPtr)
            
            # Get normals. # The number of vertices on every polygon is expected
            # to be 3.
            
            n0 = OpenMaya.MVector()
            n1 = OpenMaya.MVector()
            n2 = OpenMaya.MVector()
            
            faceIt.getNormal(0, n0)
            faceIt.getNormal(1, n1)
            faceIt.getNormal(2, n2)
            
            faceIt.getVertices(faceVertsArray)
            
            vel0, vel1, vel2 = self.getVelocities(fnMesh, list(faceVertsArray))
            
            # Get UVs.
            
            faceIt.getUV(0, float2Ptr)
            u0 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 0)
            v0 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 1)
            faceIt.getUV(1, float2Ptr)
            u1 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 0)
            v1 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 1)
            faceIt.getUV(2, float2Ptr)
            u2 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 0)
            v2 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 1)
            
            # Compute the coefficient for the normal basis function using the
            # finite element method.
            
            cnx0 = n0.x
            cnx1 = n0.x - n1.x
            cnx2 = n0.x - n2.x
            
            cny0 = n0.y
            cny1 = n0.y - n1.y
            cny2 = n0.y - n2.y
            
            cnz0 = n0.z
            cnz1 = n0.z - n1.z
            cnz2 = n0.z - n2.z
            
            # Compute the Jacobian of the normal with respect to UV coordinates
            # using this basis function.
            
            dnx_ds = cnx1
            dnx_dt = cnx2
            dny_ds = cny1
            dny_dt = cny2
            dnz_ds = cnz1
            dnz_dt = cnz2
            
            ds_du = 1.0 / (u1 - u0) if (u1 - u0) else 0
            ds_dv = 1.0 / (v1 - v0) if (v1 - v0) else 0
            dt_du = 1.0 / (u2 - u0) if (u2 - u0) else 0
            dt_dv = 1.0 / (v2 - v0) if (v2 - v0) else 0
            
            dnx_du = dnx_ds * ds_du + dnx_dt * dt_du
            dnx_dv = dnx_ds * ds_dv + dnx_dt * dt_dv
            dny_du = dny_ds * ds_du + dny_dt * dt_du
            dny_dv = dny_ds * ds_dv + dny_dt * dt_dv
            dnz_du = dnz_ds * ds_du + dnz_dt * dt_du
            dnz_dv = dnz_ds * ds_dv + dnz_dt * dt_dv
            
            jacob_n = [[dnx_du, dnx_dv], [dny_du, dny_dv], [dnz_du, dnz_dv]]
            
            # Compute the curvature term in the sizing field.
            m_curvature = M.multmm(M.transposem(jacob_n), jacob_n)
            
            # Compute the coefficient for the velocity basis function using the
            # finite element method.
            
            cvx0 = vel0[0]
            cvx1 = vel0[0] - vel1[0]
            cvx2 = vel0[0] - vel2[0]
            
            cvy0 = vel0[1]
            cvy1 = vel0[1] - vel1[1]
            cvy2 = vel0[1] - vel2[1]
            
            cvz0 = vel0[2]
            cvz1 = vel0[2] - vel1[2]
            cvz2 = vel0[2] - vel2[2]
            
            # Compute the Jacobian of the velocity with respect to UV coordinates
            # using this basis function.
            
            dvx_ds = cvx1
            dvx_dt = cvx2
            dvy_ds = cvy1
            dvy_dt = cvy2
            dvz_ds = cvz1
            dvz_dt = cvz2
            
            dvx_du = dvx_ds * ds_du + dvx_dt * dt_du
            dvx_dv = dvx_ds * ds_dv + dvx_dt * dt_dv
            dvy_du = dvy_ds * ds_du + dvy_dt * dt_du
            dvy_dv = dvy_ds * ds_dv + dvy_dt * dt_dv
            dvz_du = dvz_ds * ds_du + dvz_dt * dt_du
            dvz_dv = dvz_ds * ds_dv + dvz_dt * dt_dv
            
            jacob_v = [[dvx_du, dvx_dv], [dvy_du, dvy_dv], [dvz_du, dvz_dv]]
            
            # Compute the velocity term in the sizing field.
            m_velocity = M.multmm(M.transposem(jacob_v), jacob_v)
            
            # TODO: Compute the compression term in the sizing field.
            
            # Add the terms up to get the sizing field of this face.
            # Since this sizing field changes every frame, we do not write it
            # to per-face blind data.
            ffields[i] = M.addm(m_curvature, m_velocity)
        
        return ffields
    
    
    # Compute sizing fields of given vertices. The sizing field of a vertex is
    # computed by averaging the sizing fields of faces connected to it weighed
    # by respective face areas.
    
    def computeVertexSizingFields(self, fnMesh, verts = [], ffields = []):
        
        # To avoid repetitive computation of face sizing fields,
        # and the disruption of face indices caused by edge manipulation,
        # take the pre-computed face sizing field as an argument.
        
        vertIt = OpenMaya.MItMeshVertex(self.currentMesh)
        faceIt = OpenMaya.MItMeshPolygon(self.currentMesh)
        
        utilIdx = OpenMaya.MScriptUtil()
        utilIdx.createFromInt(0)
        intPtr = utilIdx.asIntPtr()
        
        utilArea = OpenMaya.MScriptUtil()
        utilArea.createFromDouble(0.0)
        doublePtr = utilArea.asDoublePtr()
        
        vfields = [None] * len(verts)
        
        for i in range(0, len(verts)):
            
            vert = verts[i]
            
            vertIt.setIndex(vert, intPtr)
            
            connectedFacesArray = OpenMaya.MIntArray()
            vertIt.getConnectedFaces(connectedFacesArray)
            connectedFaces = list(connectedFacesArray)
            
            ffieldSum = [[0.0, 0.0], [0.0, 0.0]]
            areaSum = 0.0
            
            for face in connectedFaces:
                
                if ffields:
                    ffield = ffields[face]
                else:
                    ffield = self.getFaceSizingFields(fnMesh, [face])[0]
                    pass
                
                faceIt.setIndex(face, intPtr)
                faceIt.getArea(doublePtr)
                area = OpenMaya.MScriptUtil.getDouble(doublePtr)
                
                # Accumulate connected face sizing fields weighed by
                # respective face areas.
                ffieldSum = M.addm(ffieldSum, M.multms(ffield, area))
                areaSum += area
            
            # Average the face sizing fields.
            if areaSum > 0:
                vfields[i] = M.divms(ffieldSum, areaSum)
        
        return vfields
    
    
    # Compute the sizes of given edges based on sizing fields. To improve
    # efficiency, the squares of the sizes are actually returned.
    
    def computeEdgeSizes(self, fnMesh, vfields, edges):
        
        utilInt2 = OpenMaya.MScriptUtil()
        utilInt2.createFromInt(0, 0)
        int2Ptr = utilInt2.asInt2Ptr()
        
        utilInt = OpenMaya.MScriptUtil()
        utilInt.createFromInt(0)
        intPtr = utilInt.asIntPtr()
        
        esize2s = [0.0] * len(edges)
        
        vertIt = OpenMaya.MItMeshVertex(self.currentMesh)
        
        uArray = OpenMaya.MFloatArray()
        vArray = OpenMaya.MFloatArray()
        faceIdsArray = OpenMaya.MIntArray()
        
        for i in range(0, len(edges)):
            
            edge = edges[i]
            
            # Get indices of the two end points of the edge.
            fnMesh.getEdgeVertices(edge, int2Ptr)
            vert0 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 0)
            vert1 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 1)
            
            # Get end point sizing fields.
            vfield0 = vfields[vert0]
            vfield1 = vfields[vert1]
            
            # Get end point UV coordinates.
            
            vertIt.setIndex(vert0, intPtr)
            vertIt.getUVs(uArray, vArray, faceIdsArray)
            u0 = uArray[0]
            v0 = uArray[0]
            
            vertIt.setIndex(vert1, intPtr)
            vertIt.getUVs(uArray, vArray, faceIdsArray)
            u1 = uArray[0]
            v1 = uArray[0]
            
            # Compute the average of the two vertex sizing field.
            avgField = M.multms(M.addm(vfield0, vfield1), 0.5)
            uv0 = M.column([u0, v0])
            uv1 = M.column([u1, v1])
            uv01 = M.subm(uv0, uv1)
            
            # Compute the edge size square.
            esize2 = M.multmm(M.multmm(M.transposem(uv01), avgField), uv01)
            esize2s[i] = esize2[0][0]
        
        return esize2s
    
    
    # Compute the ratios of lengths in world coordinates and UV coordinates at
    # current time step for given edges.
    
    def computeWUvs(self, fnMesh, edges = []):
        
        utilInt2 = OpenMaya.MScriptUtil()
        utilInt2.createFromInt(0, 0)
        int2Ptr = utilInt2.asInt2Ptr()
        
        utilInt = OpenMaya.MScriptUtil()
        utilInt.createFromInt(0, 0)
        intPtr = utilInt.asIntPtr()
        
        utilUv = OpenMaya.MScriptUtil()
        utilUv.createFromDouble(0.0, 0.0)
        float2Ptr = utilUv.asFloat2Ptr()
        
        utilLength = OpenMaya.MScriptUtil()
        utilLength.createFromDouble(0.0)
        doublePtr = utilLength.asDoublePtr()
        
        vertIt = OpenMaya.MItMeshVertex(self.currentMesh)
        edgeIt = OpenMaya.MItMeshEdge(self.currentMesh)
        
        wuvs = [0.0] * len(edges)
        
        for i in range(0, len(edges)):
            
            edge = edges[i]
            
            edgeIt.setIndex(edge, intPtr)
            
            # Get the end points of the edge.
            fnMesh.getEdgeVertices(edge, int2Ptr)
            vert0 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 0)
            vert1 = OpenMaya.MScriptUtil.getInt2ArrayItem(int2Ptr, 0, 1)
            
            # Get the UV coordinates of the end points.
            
            vertIt.setIndex(vert0, intPtr)
            vertIt.getUV(float2Ptr)
            u0 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 0)
            v0 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 1)
            
            vertIt.setIndex(vert1, intPtr)
            vertIt.getUV(float2Ptr)
            u1 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 0)
            v1 = OpenMaya.MScriptUtil.getFloat2ArrayItem(float2Ptr, 0, 1)
            
            # Compute edge length in UV coordinates.
            lenUv = M.frobNorm(M.column([u1 - u0, v1 - v0]))
            # Get the length of the edge in world coordinates.
            edgeIt.getLength(doublePtr)
            lenW = OpenMaya.MScriptUtil.getDouble(doublePtr)
            
            # Compute the ratio of the lengths in different coordinates.
            wuvs[i] = lenW / lenUv
            
        return wuvs
    
    
    # The main evaluation logic of the dependency node.
    
    def compute(self, plug, dataBlock):
        
        if plug == AdaptiveRemeshClothNode.outputMeshAttr:
            
            currentTime = dataBlock.inputValue(AdaptiveRemeshClothNode.currentTimeAttr).asTime();
            inMeshDataHandle = dataBlock.inputValue(AdaptiveRemeshClothNode.inputMeshAttr)
            
            # Reset the simulation when the playback time is set to the
            # start.
            if currentTime.value() <= 1.0:
                
                # Reset the mesh by duplicating mesh from inputMesh attribute.
                inFnMeshData = inMeshDataHandle.asMesh()
                fnMesh = OpenMaya.MFnMesh(inFnMeshData)
                fnMesh.copy(inFnMeshData, self.currentMesh)
                fnMesh.setObject(self.currentMesh)
                
                # Initialize blind data templates and attribute values.
                self.initializeBlindDataTypes(fnMesh)
                self.initializeBlindData(fnMesh,
                                         list(range(0, fnMesh.numVertices())),
                                         list(range(0, fnMesh.numEdges())),
                                         list(range(0, fnMesh.numPolygons())))
                
                # self.constraintVerts = [0, fnMesh.numVertices() - 1]
                
                constraints = [True] * len(self.constraintVerts)
                self.setVertexConstraints(fnMesh, self.constraintVerts, constraints)
                
                # Compute the world-UV coordinate ratios at the beginning of
                # the simulation and its average.
                self.wuvs = self.computeWUvs(fnMesh, list(range(0, fnMesh.numEdges())))
                self.avgWUv = sum(self.wuvs) / len(self.wuvs)
                
                print(self.kPluginNodeName + ": Internal mesh reset.")
            
            # Otherwise, step the simulation.
            else:
                
                time0 = os.times()[0]
                
                fnMesh = OpenMaya.MFnMesh(self.currentMesh)
                
                # Step the basic cloth simulation cycle.
                # Actually we can do remeshing every few time steps.
                self.step(fnMesh, self.timeStep)
                
                timeNoRemesh = os.times()[0]
                
                # Compute the vertex fields of faces and vertices and edge
                # sizes based on them. Then split all edges with sufficiently
                # large sizes.
                
                ffields = self.computeFaceSizingFields(fnMesh, list(range(0, fnMesh.numPolygons())))
                vfields = self.computeVertexSizingFields(fnMesh, list(range(0, fnMesh.numVertices())), ffields)
                esize2s = self.computeEdgeSizes(fnMesh, vfields, list(range(0, fnMesh.numEdges())))
                      
                self.splitEdges(fnMesh, ffields, vfields, esize2s)
                
                # Collapse edges as many as possible.
                # TODO: Currently commented out because it is too slow.
                # self.collapseEdges(fnMesh)
                
                timeRemesh = os.times()[0]
                
                print("Basic cloth simulation cycle: {}".format(timeNoRemesh - time0))
                print("Remeshing: {}".format(timeRemesh - timeNoRemesh))
                print("--------------------------------")
                
            # Put the computed data on the plug.
            outMeshDataHandle = dataBlock.outputValue(AdaptiveRemeshClothNode.outputMeshAttr)
            outMeshDataHandle.setMObject(self.currentMesh)
            
            # Indicate the data requested on the plug has been updated.
            dataBlock.setClean(plug)
            
        else:
            
            return OpenMaya.kUnknownParameter
    

# Initialize the dependency node plugin.

def initializePlugin(obj):
    
    plugin = OpenMayaMPx.MFnPlugin(
        obj,
        "Dale Zhao",
        "0.01",
        "Any"
    )
    
    try:
        
        print(AdaptiveRemeshClothNode.kPluginNodeName + ": Registering...")
        plugin.registerNode(
            AdaptiveRemeshClothNode.kPluginNodeName,
            AdaptiveRemeshClothNode.kPluginNodeId,
            AdaptiveRemeshClothNode.nodeCreator,
            AdaptiveRemeshClothNode.nodeInitializer
        )
        print(AdaptiveRemeshClothNode.kPluginNodeName + ": Registered.")
    
    except:
        
        print(AdaptiveRemeshClothNode.kPluginNodeName + ": Failed to register.")
        raise Exception(AdaptiveRemeshClothNode.kPluginNodeName + ": Failed to register.")


# Uninitialize the dependency node plugin.

def uninitializePlugin(obj):
    
    plugin = OpenMayaMPx.MFnPlugin(obj)
    
    try:
        
        print(AdaptiveRemeshClothNode.kPluginNodeName + ": Deregistering...")
        plugin.deregisterNode(AdaptiveRemeshClothNode.kPluginNodeId)
        print(AdaptiveRemeshClothNode.kPluginNodeName + ": Deregistered.")
    
    except:
        
        print(AdaptiveRemeshClothNode.kPluginNodeName + ": Failed to deregister.")
        raise Exception(AdaptiveRemeshClothNode.kPluginNodeName + ": Failed to deregister.")
    
    
