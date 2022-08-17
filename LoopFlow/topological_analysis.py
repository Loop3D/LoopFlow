# -*- coding: utf-8 -*-
"""
Updated on Wednesday 17/08/2022

Comptutation of the topological graph from regular grid voxets of geological models 
and associated fault topology matrix. 
Identify and create interformation and fault nodes.
Identify edges between adjacent nodes (contact on a voxel face (in the 3D case) 
                                       or edge (in the 2D case))

@author: Guillaume PIROT
"""
__version__="0.1.18"

import numpy as np
import pandas as pd
import networkx as nx
import sys
import os
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

# INPUT VARIABLES
# nd_X: regular grid ND-array of x-coordinates
# nd_Y: regular grid ND-array of y-coordinates
# nd_Z: regular grid ND-array of z-coordinates
# nd_lithocodes: regular grid ND-array of lithocode (geological unit id number)
# nd_topo_faults: regular grid ND+1-array of topological matrix of faults (last dimension: fault dimension)
# fault_names: array of fault names
# destination: location and filename without extension to export a GML graph, and dataframes of nodes and edges
# verb=False: verbose option for memory analysis or debugging

# INNER VARIABLE TO CHECK MEMORY USAGE WHEN VERB==TRUE
mem_unit = 'MB'
mem_factor = 1E6

# connectFaces = np.asarray([[1,0,0],
#                            [0,1,0],
#                            [0,0,1]])
# connectEdges = np.asarray([[ 1, 1, 0],
#                            [ 1,-1, 0],
#                            [ 1, 0, 1],
#                            [ 1, 0,-1],
#                            [ 0, 1, 1],
#                            [ 0, 1,-1]])
# connectCrnrs = np.asarray([[ 1, 1, 1],
#                            [-1, 1, 1],
#                            [-1,-1, 1],
#                            [ 1,-1, 1]])

# OUTPUT VARIABLES
# df_nodes: graph nodes described as a dataframe
# df_edges: graph edges described as a dataframe
# G: graph built with networkx  

def reggrid_topology_graph(nd_X,nd_Y,nd_Z,nd_lithocodes,nd_topo_faults,fault_names,unique_edges=True,simplify=True,verb=False):
    print('Running topological_analysis version '+ __version__)
    dim = nd_lithocodes.shape
    ndim=len(dim)
    nbfaults = len(fault_names)
    dim0 = (dim[0]-1,dim[1],dim[2])
    dim1 = (dim[0],dim[1]-1,dim[2])
    dim2 = (dim[0],dim[1],dim[2]-1)
    mx_internodes_dim0 = np.zeros(dim0).astype(int).flatten()
    mx_internodes_dim1 = np.zeros(dim1).astype(int).flatten()
    mx_internodes_dim2 = np.zeros(dim2).astype(int).flatten()
    # GET dx, dy, dz FROM DATA
    dx = np.unique(np.diff(np.sort(np.unique(nd_X.flatten()))))
    dy = np.unique(np.diff(np.sort(np.unique(nd_Y.flatten()))))
    dz = np.unique(np.diff(np.sort(np.unique(nd_Z.flatten()))))
    if np.size(dx)>0:
        dx=dx[0]
    else:
        dx=1            
    if np.size(dy)>0:
        dy=dy[0]
    else:
        dy=1
    if np.size(dz)>0:
        dz=dz[0]
    else:
        dz=1
    
    # VOXEL ID NUMBERING
    nd_voxelid = np.reshape(np.arange(np.prod(dim)),dim)
    flat_voxelid = nd_voxelid.flatten()
    ix_sorted = np.argsort(flat_voxelid)
    #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
    
    # FORMATION NODES
    df_nodes = pd.DataFrame({'id':flat_voxelid[ix_sorted],'X':nd_X.flatten()[ix_sorted],'Y':nd_Y.flatten()[ix_sorted],'Z':nd_Z.flatten()[ix_sorted],'geocode':nd_lithocodes.flatten()[ix_sorted]})
    df_nodes['description']='geological formation'
    df_nodes['orthodim']=-1 #np.nan
    
    # INITIALIZE EMPTY EDGE DATAFRAME
    df_edges = pd.DataFrame(columns=['id_node_src','id_node_tgt','type','label']) #,'geocode_src','geocode_tgt'
    
    # CURRENT NODE ID TO INCREMENT WHEN ADDING NODES
    cur_node_id = flat_voxelid[ix_sorted][-1]+1
    
    if verb==True:
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE - INIT')
        print('nd_lithocodes: ' + str(sys.getsizeof(nd_lithocodes)/mem_factor) + ' ' + mem_unit)
        print('nd_X: ' + str(sys.getsizeof(nd_X)/mem_factor) + ' ' + mem_unit)
        print('nd_Y: ' + str(sys.getsizeof(nd_Y)/mem_factor) + ' ' + mem_unit)
        print('nd_Z: ' + str(sys.getsizeof(nd_Z)/mem_factor) + ' ' + mem_unit)
        print('nd_topo_faults: ' + str(sys.getsizeof(nd_topo_faults)/mem_factor) + ' ' + mem_unit)
        print('fault_names: ' + str(sys.getsizeof(fault_names)/mem_factor) + ' ' + mem_unit)
        print('nd_voxelid: ' + str(sys.getsizeof(nd_voxelid)/mem_factor) + ' ' + mem_unit)
        print('flat_voxelid: ' + str(sys.getsizeof(flat_voxelid)/mem_factor) + ' ' + mem_unit)
        print('ix_sorted: ' + str(sys.getsizeof(ix_sorted)/mem_factor) + ' ' + mem_unit)

    # OVERLAPPNG TEST ALONG EACH DIMENSION
    for d in range(ndim):
        tmpID1 = nd_voxelid.take(indices=range(1, dim[d]), axis=d).flatten()
        tmpID2 = nd_voxelid.take(indices=range(0, dim[d]-1), axis=d).flatten()
        tmpX = (1/2 * (nd_X.take(indices=range(1, dim[d]), axis=d) + nd_X.take(indices=range(0, dim[d]-1), axis=d)) ).flatten()
        tmpY = (1/2 * (nd_Y.take(indices=range(1, dim[d]), axis=d) + nd_Y.take(indices=range(0, dim[d]-1), axis=d)) ).flatten()
        tmpZ = (1/2 * (nd_Z.take(indices=range(1, dim[d]), axis=d) + nd_Z.take(indices=range(0, dim[d]-1), axis=d)) ).flatten()
        tmpNoIN = np.zeros(tmpX.shape)==1 # TO PREVENT INTERFORMATION NODES IF THERE IS A FAULT 
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - dim '+str(d) +' fault nodes.')
            print('tmpID1: ' + str(sys.getsizeof(tmpID1)/mem_factor) + ' ' + mem_unit)
            print('tmpID2: ' + str(sys.getsizeof(tmpID2)/mem_factor) + ' ' + mem_unit)
            print('tmpX: ' + str(sys.getsizeof(tmpX)/mem_factor) + ' ' + mem_unit)
            print('tmpY: ' + str(sys.getsizeof(tmpY)/mem_factor) + ' ' + mem_unit)
            print('tmpZ: ' + str(sys.getsizeof(tmpZ)/mem_factor) + ' ' + mem_unit)
        # AND ALSO FORBID INTRAFORMATION EDGE
        # Fault Nodes + fault-formation edges
        for f in range(nbfaults):
            cur_topo_flt = nd_topo_faults.take(indices=f, axis=ndim)
            cur_flt_name = fault_names[f]
            tmpF = ((cur_topo_flt.take(indices=range(1, dim[d]), axis=d) * cur_topo_flt.take(indices=range(0, dim[d]-1), axis=d))==-1 ).flatten()
            tmpNoIN = (tmpNoIN | tmpF) # TO FORBID INTRAFORMATION EDGE
            ixfn = np.asarray(np.where(tmpF==True)).flatten()
            if len(ixfn)>0:
                # FAULT NODES
                new_node_id = np.arange(cur_node_id,cur_node_id+len(ixfn))
                df_tmp = pd.DataFrame({'id':new_node_id,'X':tmpX[ixfn],'Y':tmpY[ixfn],'Z':tmpZ[ixfn]})
                df_tmp['geocode'] = cur_flt_name
                df_tmp['description'] = 'fault node'
                df_tmp['orthodim'] = d
                cur_node_id = cur_node_id+len(ixfn)
                df_nodes= pd.concat([df_nodes,df_tmp]) 
                df_nodes.reset_index(drop=True, inplace=True)
                # add new node id in corresponding dim internode matrix
                if d==0:
                    mx_internodes_dim0[ixfn] = new_node_id
                elif d==1:
                    mx_internodes_dim1[ixfn] = new_node_id
                elif d==2:
                    mx_internodes_dim2[ixfn] = new_node_id
                # FAULT FORMATION EDGES
                #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
                lab1 = np.asarray(df_tmp['geocode']+' - geol_' + (df_nodes.loc[tmpID1[ixfn],'geocode'].values).astype(str))
                df_tmpEdges1 = pd.DataFrame({'id_node_src':new_node_id,'id_node_tgt':tmpID1[ixfn],'type':'fault-formation','label':lab1})
                lab2 = np.asarray(df_tmp['geocode']+' - geol_' + (df_nodes.loc[tmpID2[ixfn],'geocode'].values).astype(str))
                df_tmpEdges2 = pd.DataFrame({'id_node_src':new_node_id,'id_node_tgt':tmpID2[ixfn],'type':'fault-formation','label':lab2})
                df_edges = pd.concat([df_edges,df_tmpEdges1,df_tmpEdges2]) 
                if verb==True:
                    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - dim '+str(d)+' '+cur_flt_name+'-formation edges')
                    print('tmpNoIN: ' + str(sys.getsizeof(tmpNoIN)/mem_factor) + ' ' + mem_unit)
                    print('lab1: ' + str(sys.getsizeof(lab1)/mem_factor) + ' ' + mem_unit)
                    print('lab2: ' + str(sys.getsizeof(lab2)/mem_factor) + ' ' + mem_unit)
                    print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
                    print('df_tmpEdges2: ' + str(sys.getsizeof(df_tmpEdges2)/mem_factor) + ' ' + mem_unit)
                del df_tmp,lab1,lab2,df_tmpEdges1,df_tmpEdges2
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - dim '+str(d) + ' inter nodes...')
            print('tmpNoIN: ' + str(sys.getsizeof(tmpNoIN)/mem_factor) + ' ' + mem_unit)
        # Interformation Nodes + edges + intraedges
        tmpI = ((nd_lithocodes.take(indices=range(1, dim[d]), axis=d) - nd_lithocodes.take(indices=range(0, dim[d]-1), axis=d))!=0 ).flatten()
        ixin = np.asarray(np.where(((tmpI==True)&(tmpNoIN==False)))).flatten()
        if len(ixin)>0:
            # INTERFORMATION NODES
            new_node_id = np.arange(cur_node_id,cur_node_id+len(ixin))
            df_tmp = pd.DataFrame({'id':new_node_id,'X':tmpX[ixin],'Y':tmpY[ixin],'Z':tmpZ[ixin]})
            geocode_pair = np.zeros((len(ixin),2)).astype(int)
            geocode_pair[:,0]=df_nodes.loc[tmpID1[ixin],'geocode'].values
            geocode_pair[:,1]=df_nodes.loc[tmpID2[ixin],'geocode'].values
            geocode_pair = np.sort(geocode_pair,axis=1)
            df_pair = pd.DataFrame(geocode_pair)
            df_tmp['geocode'] = 'inter-geol_'+(df_pair.iloc[:,0]).astype(str) +'-geol_'+(df_pair.iloc[:,1]).astype(str)
            df_tmp['description'] = 'interformation node'
            df_tmp['orthodim'] = d
            df_tmp['interform_geocode_a'] = geocode_pair[:,0]
            df_tmp['interform_geocode_b'] = geocode_pair[:,1] 
            cur_node_id = cur_node_id+len(ixin)
            df_nodes= pd.concat([df_nodes,df_tmp])
            df_nodes.reset_index(drop=True, inplace=True)
            # add new node id in corresponding dim internode matrix
            if d==0:
                mx_internodes_dim0[ixin] = new_node_id
            elif d==1:
                mx_internodes_dim1[ixin] = new_node_id
            elif d==2:
                mx_internodes_dim2[ixin] = new_node_id
            # INTERFORMATION - FORMATION EDGES
            #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
            df_tmpEdges1 = pd.DataFrame({'id_node_src':new_node_id,'id_node_tgt':tmpID1[ixin],'type':'interform-formation'})
            df_tmpEdges1['label'] = df_tmp['geocode'].values+' - geol_' + (df_nodes.loc[tmpID1[ixin],'geocode'].astype(str)).values
            df_tmpEdges2 = pd.DataFrame({'id_node_src':new_node_id,'id_node_tgt':tmpID2[ixin],'type':'interform-formation'})
            df_tmpEdges2['label'] = df_tmp['geocode'].values+' - geol_' + (df_nodes.loc[tmpID2[ixin],'geocode'].astype(str)).values
            df_edges = pd.concat([df_edges,df_tmpEdges1,df_tmpEdges2]) 
            if verb==True:
                print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - dim '+str(d)+' interform nodes + interform-form edges')
                print('df_tmp(nodes): ' + str(sys.getsizeof(df_tmp)/mem_factor) + ' ' + mem_unit)
                print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
                print('df_tmpEdges2: ' + str(sys.getsizeof(df_tmpEdges2)/mem_factor) + ' ' + mem_unit)
            del df_tmp,df_tmpEdges1,df_tmpEdges2
        # Intraformation edges - voxel face connections
        ixintra = np.asarray(np.where(((tmpI==False)&(tmpNoIN==False)))).flatten()
        if len(ixintra)>0:
            #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
            df_tmpEdges1 = pd.DataFrame({'id_node_src':tmpID1[ixintra],'id_node_tgt':tmpID2[ixintra],'type':'intra-formation'})
            df_tmpEdges1['label'] = 'geol_' + (df_nodes.loc[tmpID1[ixintra],'geocode'].astype(str)).values
            df_edges = pd.concat([df_edges,df_tmpEdges1]) 
            if verb==True:
                print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - dim '+str(d)+' intraformation edges')
                print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
            del df_tmpEdges1
    # ------------------------------------------------------------------------
    # Intraformation edges - voxel edge (intra-formation) connections
    # ------------------------------------------------------------------------
    # Edge pair 1/6
    tmpID1 = nd_voxelid.take(indices=range(1, dim[0]), axis=0).take(indices=range(1, dim[1]), axis=1).flatten()
    tmpID2 = nd_voxelid.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(0, dim[1]-1), axis=1).flatten()
    tmpLC1 = nd_lithocodes.take(indices=range(1, dim[0]), axis=0).take(indices=range(1, dim[1]), axis=1).flatten()
    tmpLC2 = nd_lithocodes.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(0, dim[1]-1), axis=1).flatten()
    tmpix = np.asarray(np.where(tmpLC1==tmpLC2)).flatten()
    if len(tmpix)>0:
        #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
        df_tmpEdges1 = pd.DataFrame({'id_node_src':tmpID1[tmpix],'id_node_tgt':tmpID2[tmpix],'type':'intra-formation'})
        df_tmpEdges1['label'] = 'geol_' + (df_nodes.loc[tmpID1[tmpix],'geocode'].astype(str)).values
        df_tmpEdges1['diag'] = 1
        df_edges = pd.concat([df_edges,df_tmpEdges1]) 
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - voxel edge (intra-formation) connections 1/6')
            print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
        del df_tmpEdges1
    # Edge pair 2/6
    tmpID1 = nd_voxelid.take(indices=range(1, dim[0]), axis=0).take(indices=range(0, dim[1]-1), axis=1).flatten()
    tmpID2 = nd_voxelid.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(1, dim[1]), axis=1).flatten()
    tmpLC1 = nd_lithocodes.take(indices=range(1, dim[0]), axis=0).take(indices=range(0, dim[1]-1), axis=1).flatten()
    tmpLC2 = nd_lithocodes.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(1, dim[1]), axis=1).flatten()
    tmpix = np.asarray(np.where(tmpLC1==tmpLC2)).flatten()
    if len(tmpix)>0:
        #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
        df_tmpEdges1 = pd.DataFrame({'id_node_src':tmpID1[tmpix],'id_node_tgt':tmpID2[tmpix],'type':'intra-formation'})
        df_tmpEdges1['label'] = 'geol_' + (df_nodes.loc[tmpID1[tmpix],'geocode'].astype(str)).values
        df_tmpEdges1['diag'] = 1
        df_edges = pd.concat([df_edges,df_tmpEdges1]) 
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - voxel edge (intra-formation) connections 2/6')
            print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
        del df_tmpEdges1
    # Edge pair 3/6
    tmpID1 = nd_voxelid.take(indices=range(1, dim[0]), axis=0).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpID2 = nd_voxelid.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpLC1 = nd_lithocodes.take(indices=range(1, dim[0]), axis=0).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpLC2 = nd_lithocodes.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpix = np.asarray(np.where(tmpLC1==tmpLC2)).flatten()
    if len(tmpix)>0:
        #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
        df_tmpEdges1 = pd.DataFrame({'id_node_src':tmpID1[tmpix],'id_node_tgt':tmpID2[tmpix],'type':'intra-formation'})
        df_tmpEdges1['label'] = 'geol_' + (df_nodes.loc[tmpID1[tmpix],'geocode'].astype(str)).values
        df_tmpEdges1['diag'] = 1
        df_edges = pd.concat([df_edges,df_tmpEdges1]) 
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - voxel edge (intra-formation) connections 3/6')
            print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
        del df_tmpEdges1
    # Edge pair 4/6
    tmpID1 = nd_voxelid.take(indices=range(1, dim[0]), axis=0).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpID2 = nd_voxelid.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpLC1 = nd_lithocodes.take(indices=range(1, dim[0]), axis=0).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpLC2 = nd_lithocodes.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpix = np.asarray(np.where(tmpLC1==tmpLC2)).flatten()
    if len(tmpix)>0:
        #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
        df_tmpEdges1 = pd.DataFrame({'id_node_src':tmpID1[tmpix],'id_node_tgt':tmpID2[tmpix],'type':'intra-formation'})
        df_tmpEdges1['label'] = 'geol_' + (df_nodes.loc[tmpID1[tmpix],'geocode'].astype(str)).values
        df_tmpEdges1['diag'] = 1
        df_edges = pd.concat([df_edges,df_tmpEdges1]) 
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - voxel edge (intra-formation) connections 4/6')
            print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
        del df_tmpEdges1
    # Edge pair 5/6
    tmpID1 = nd_voxelid.take(indices=range(1, dim[1]), axis=1).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpID2 = nd_voxelid.take(indices=range(0, dim[1]-1), axis=1).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpLC1 = nd_lithocodes.take(indices=range(1, dim[1]), axis=1).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpLC2 = nd_lithocodes.take(indices=range(0, dim[1]-1), axis=1).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpix = np.asarray(np.where(tmpLC1==tmpLC2)).flatten()
    if len(tmpix)>0:
        #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
        df_tmpEdges1 = pd.DataFrame({'id_node_src':tmpID1[tmpix],'id_node_tgt':tmpID2[tmpix],'type':'intra-formation'})
        df_tmpEdges1['label'] = 'geol_' + (df_nodes.loc[tmpID1[tmpix],'geocode'].astype(str)).values
        df_tmpEdges1['diag'] = 1
        df_edges = pd.concat([df_edges,df_tmpEdges1]) 
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - voxel edge (intra-formation) connections 5/6')
            print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
        del df_tmpEdges1
    # Edge pair 6/6
    tmpID1 = nd_voxelid.take(indices=range(1, dim[1]), axis=1).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpID2 = nd_voxelid.take(indices=range(0, dim[1]-1), axis=1).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpLC1 = nd_lithocodes.take(indices=range(1, dim[1]), axis=1).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpLC2 = nd_lithocodes.take(indices=range(0, dim[1]-1), axis=1).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpix = np.asarray(np.where(tmpLC1==tmpLC2)).flatten()
    if len(tmpix)>0:
        #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
        df_tmpEdges1 = pd.DataFrame({'id_node_src':tmpID1[tmpix],'id_node_tgt':tmpID2[tmpix],'type':'intra-formation'})
        df_tmpEdges1['label'] = 'geol_' + (df_nodes.loc[tmpID1[tmpix],'geocode'].astype(str)).values
        df_tmpEdges1['diag'] = 1
        df_edges = pd.concat([df_edges,df_tmpEdges1]) 
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - voxel edge (intra-formation) connections 6/6')
            print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
        del df_tmpEdges1
    # ------------------------------------------------------------------------
    # Intraformation edges - voxel corner (intra-formation) connections
    # ------------------------------------------------------------------------
    # Corner pair 1/4
    tmpID1 = nd_voxelid.take(indices=range(1, dim[0]), axis=0).take(indices=range(1, dim[1]), axis=1).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpID2 = nd_voxelid.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(0, dim[1]-1), axis=1).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpLC1 = nd_lithocodes.take(indices=range(1, dim[0]), axis=0).take(indices=range(1, dim[1]), axis=1).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpLC2 = nd_lithocodes.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(0, dim[1]-1), axis=1).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpix = np.asarray(np.where(tmpLC1==tmpLC2)).flatten()
    if len(tmpix)>0:
        #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
        df_tmpEdges1 = pd.DataFrame({'id_node_src':tmpID1[tmpix],'id_node_tgt':tmpID2[tmpix],'type':'intra-formation'})
        df_tmpEdges1['label'] = 'geol_' + (df_nodes.loc[tmpID1[tmpix],'geocode'].astype(str)).values
        df_tmpEdges1['diag'] = 1
        df_edges = pd.concat([df_edges,df_tmpEdges1]) 
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - voxel corner (intra-formation) connections 1/4')
            print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
        del df_tmpEdges1
    # Corner pair 2/4
    tmpID1 = nd_voxelid.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(1, dim[1]), axis=1).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpID2 = nd_voxelid.take(indices=range(1, dim[0]), axis=0).take(indices=range(0, dim[1]-1), axis=1).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpLC1 = nd_lithocodes.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(1, dim[1]), axis=1).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpLC2 = nd_lithocodes.take(indices=range(1, dim[0]), axis=0).take(indices=range(0, dim[1]-1), axis=1).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpix = np.asarray(np.where(tmpLC1==tmpLC2)).flatten()
    if len(tmpix)>0:
        #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
        df_tmpEdges1 = pd.DataFrame({'id_node_src':tmpID1[tmpix],'id_node_tgt':tmpID2[tmpix],'type':'intra-formation'})
        df_tmpEdges1['label'] = 'geol_' + (df_nodes.loc[tmpID1[tmpix],'geocode'].astype(str)).values
        df_tmpEdges1['diag'] = 1
        df_edges = pd.concat([df_edges,df_tmpEdges1]) 
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - voxel corner (intra-formation) connections 2/4')
            print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
        del df_tmpEdges1
    # Corner pair 3/4
    tmpID1 = nd_voxelid.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(0, dim[1]-1), axis=1).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpID2 = nd_voxelid.take(indices=range(1, dim[0]), axis=0).take(indices=range(1, dim[1]), axis=1).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpLC1 = nd_lithocodes.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(0, dim[1]-1), axis=1).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpLC2 = nd_lithocodes.take(indices=range(1, dim[0]), axis=0).take(indices=range(1, dim[1]), axis=1).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpix = np.asarray(np.where(tmpLC1==tmpLC2)).flatten()
    if len(tmpix)>0:
        #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
        df_tmpEdges1 = pd.DataFrame({'id_node_src':tmpID1[tmpix],'id_node_tgt':tmpID2[tmpix],'type':'intra-formation'})
        df_tmpEdges1['label'] = 'geol_' + (df_nodes.loc[tmpID1[tmpix],'geocode'].astype(str)).values
        df_tmpEdges1['diag'] = 1
        df_edges = pd.concat([df_edges,df_tmpEdges1]) 
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - voxel corner (intra-formation) connections 3/4')
            print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
        del df_tmpEdges1
    # Corner pair 4/4
    tmpID1 = nd_voxelid.take(indices=range(1, dim[0]), axis=0).take(indices=range(0, dim[1]-1), axis=1).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpID2 = nd_voxelid.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(1, dim[1]), axis=1).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpLC1 = nd_lithocodes.take(indices=range(1, dim[0]), axis=0).take(indices=range(0, dim[1]-1), axis=1).take(indices=range(1, dim[2]), axis=2).flatten()
    tmpLC2 = nd_lithocodes.take(indices=range(0, dim[0]-1), axis=0).take(indices=range(1, dim[1]), axis=1).take(indices=range(0, dim[2]-1), axis=2).flatten()
    tmpix = np.asarray(np.where(tmpLC1==tmpLC2)).flatten()
    if len(tmpix)>0:
        #!! IMPORTANT : THE FOLLOWING ASSUMES index numbering = voxelid for the voxet nodes
        df_tmpEdges1 = pd.DataFrame({'id_node_src':tmpID1[tmpix],'id_node_tgt':tmpID2[tmpix],'type':'intra-formation'})
        df_tmpEdges1['label'] = 'geol_' + (df_nodes.loc[tmpID1[tmpix],'geocode'].astype(str)).values
        df_tmpEdges1['diag'] = 1
        df_edges = pd.concat([df_edges,df_tmpEdges1]) 
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - voxel corner (intra-formation) connections 4/4')
            print('df_tmpEdges1: ' + str(sys.getsizeof(df_tmpEdges1)/mem_factor) + ' ' + mem_unit)
        del df_tmpEdges1
    # ------------------------------------------------------------------------
    # REMOVE INTRA-FORMATION EDGES CROSSING FAULT
    # ------------------------------------------------------------------------
    df_edges.reset_index(inplace=True,drop=True)
    ixdiag = np.asarray(np.where(df_edges['diag']==1)).flatten()
    ix2drp = ixdiag==-1
    ixsrc = df_edges.loc[ixdiag,'id_node_src'].values.astype(int)
    ixtgt = df_edges.loc[ixdiag,'id_node_tgt'].values.astype(int)
    # for each fault, check if edge nodes are across a fault
    for f in range(nbfaults):
        tmpF = nd_topo_faults.take(indices=f, axis=ndim).flatten()
        ix2drp = ix2drp | (tmpF[ixsrc]*tmpF[ixtgt]==-1)
    df_edges.drop(ixdiag[ix2drp],inplace=True)
    df_edges.drop(columns=['diag'],inplace=True)

    # df_edges_B = df_edges.copy()

    # ------------------------------------------------------------------------
    # FAULT-FAULT, FAULT-INTERFORM & INTERFORM-INTERFORM EDGES
    # ------------------------------------------------------------------------
    # RESHAPE INTERNODE MATRICES
    mx_internodes_dim0 = np.reshape(mx_internodes_dim0,dim0)
    mx_internodes_dim1 = np.reshape(mx_internodes_dim1,dim1)
    mx_internodes_dim2 = np.reshape(mx_internodes_dim2,dim2)
    # FOR DIM0 INTERNODE MATRIX SHIFT ALONG DIM1 AND DIM2
    tmpID1,tmpID2 = extract_shift_dim(mx_internodes_dim0,1)
    df_edges = add_new_edges(tmpID1,tmpID2,df_edges,verb=verb,suffix="DIM0 INTERNODE EDGES D1")
    tmpID1,tmpID2 = extract_shift_dim(mx_internodes_dim0,2)
    df_edges = add_new_edges(tmpID1,tmpID2,df_edges,verb=verb,suffix="DIM0 INTERNODE EDGES D2")
    # FOR DIM1 INTERNODE MATRIX SHIFT ALONG DIM0 AND DIM2
    tmpID1,tmpID2 = extract_shift_dim(mx_internodes_dim1,0)
    df_edges = add_new_edges(tmpID1,tmpID2,df_edges,verb=verb,suffix="DIM1 INTERNODE EDGES D0")
    tmpID1,tmpID2 = extract_shift_dim(mx_internodes_dim1,2)
    df_edges = add_new_edges(tmpID1,tmpID2,df_edges,verb=verb,suffix="DIM1 INTERNODE EDGES D2")
    # FOR DIM2 INTERNODE MATRIX SHIFT ALONG DIM0 AND DIM1
    tmpID1,tmpID2 = extract_shift_dim(mx_internodes_dim2,1)
    df_edges = add_new_edges(tmpID1,tmpID2,df_edges,verb=verb,suffix="DIM2 INTERNODE EDGES D1")
    tmpID1,tmpID2 = extract_shift_dim(mx_internodes_dim2,0)
    df_edges = add_new_edges(tmpID1,tmpID2,df_edges,verb=verb,suffix="DIM2 INTERNODE EDGES D0")
    # FOR EACH PAIR OF INTERNODE MATRICE ALIGN THE 8 CORNERS
    # mx_internodes_dim0 - mx_internodes_dim1 edges
    if (np.prod(dim0)*np.prod(dim1))>0: 
        if verb: print("INTERNODE DIM0 - INTERNODE DIM1 EDGES")
        tmp0a,tmp0b = extract_shift_dim(mx_internodes_dim0,1)
        tmp1a,tmp1b = extract_shift_dim(mx_internodes_dim1,0)
        df_edges = add_new_edges(tmp0a,tmp1a,df_edges,verb=verb,suffix="DIM0 DIM1 INTERNODE EDGES 1/4")
        df_edges = add_new_edges(tmp0a,tmp1b,df_edges,verb=verb,suffix="DIM0 DIM1 INTERNODE EDGES 2/4")
        df_edges = add_new_edges(tmp0b,tmp1a,df_edges,verb=verb,suffix="DIM0 DIM1 INTERNODE EDGES 3/4")
        df_edges = add_new_edges(tmp0b,tmp1b,df_edges,verb=verb,suffix="DIM0 DIM1 INTERNODE EDGES 4/4")
    else:
        if verb: print("NO INTERNODE DIM0 - INTERNODE DIM1 EDGES")
    # mx_internodes_dim0 - mx_internodes_dim2 edges
    if (np.prod(dim0)*np.prod(dim2))>0: 
        if verb: print("INTERNODE DIM0 - INTERNODE DIM2 EDGES")
        tmp0a,tmp0b = extract_shift_dim(mx_internodes_dim0,2)
        tmp1a,tmp1b = extract_shift_dim(mx_internodes_dim2,0)
        df_edges = add_new_edges(tmp0a,tmp1a,df_edges,verb=verb,suffix="DIM0 DIM2 INTERNODE EDGES 1/4")
        df_edges = add_new_edges(tmp0a,tmp1b,df_edges,verb=verb,suffix="DIM0 DIM2 INTERNODE EDGES 2/4")
        df_edges = add_new_edges(tmp0b,tmp1a,df_edges,verb=verb,suffix="DIM0 DIM2 INTERNODE EDGES 3/4")
        df_edges = add_new_edges(tmp0b,tmp1b,df_edges,verb=verb,suffix="DIM0 DIM2 INTERNODE EDGES 4/4")
    else:
        if verb: print("NO INTERNODE DIM0 - INTERNODE DIM2 EDGES")
    # mx_internodes_dim1 - mx_internodes_dim2 edges
    if (np.prod(dim1)*np.prod(dim2))>0: 
        if verb: print("INTERNODE DIM1 - INTERNODE DIM2 EDGES")
        tmp0a,tmp0b = extract_shift_dim(mx_internodes_dim1,2)
        tmp1a,tmp1b = extract_shift_dim(mx_internodes_dim2,1)
        df_edges = add_new_edges(tmp0a,tmp1a,df_edges,verb=verb,suffix="DIM1 DIM2 INTERNODE EDGES 1/4")
        df_edges = add_new_edges(tmp0a,tmp1b,df_edges,verb=verb,suffix="DIM1 DIM2 INTERNODE EDGES 2/4")
        df_edges = add_new_edges(tmp0b,tmp1a,df_edges,verb=verb,suffix="DIM1 DIM2 INTERNODE EDGES 3/4")
        df_edges = add_new_edges(tmp0b,tmp1b,df_edges,verb=verb,suffix="DIM1 DIM2 INTERNODE EDGES 4/4")
    else:
        if verb: print("NO INTERNODE DIM1 - INTERNODE DIM2 EDGES")
           
    # IDENTIFY NODES WITHOUT EDGES AND IF NODES WITH SAME COORDINATES EXIST ??
    # REMOVE EVENTUAL DUPICATE EDGES
    if unique_edges == True:
        if verb:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - REMOVE DUPLICATE edges') 
        df_edges.drop_duplicates(subset=['id_node_src','id_node_tgt'], keep='first', inplace=True)
    
    # EDGE LABELS
    if verb: print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - LABELING edges')
    df_edges.reset_index(drop=True, inplace=True)
    ed_ix2label = np.asarray(np.where(df_edges["type"].isna())).flatten()
    feat1 = df_nodes.loc[df_edges.iloc[ed_ix2label,0].values,"geocode"].values
    feat2 = df_nodes.loc[df_edges.iloc[ed_ix2label,1].values,"geocode"].values
    # identify type of node same-fault, fault-fault, same-interform, interform-interform, interform-fault
    feat1flt = (np.frompyfunc(lambda x: 'fault' in x, 1, 1)(feat1)).astype(bool)
    feat2flt = (np.frompyfunc(lambda x: 'fault' in x, 1, 1)(feat2)).astype(bool)
    feat1itf = (np.frompyfunc(lambda x: 'inter' in x, 1, 1)(feat1)).astype(bool)
    feat2itf = (np.frompyfunc(lambda x: 'inter' in x, 1, 1)(feat2)).astype(bool)
    featsame = (feat1==feat2)
    ix_sf = (feat1flt & feat2flt & featsame).astype(bool)
    ix_ff = (feat1flt & feat2flt & ~featsame).astype(bool)
    ix_si = (feat1itf & feat2itf & featsame).astype(bool)
    ix_ii = (feat1itf & feat2itf & ~featsame).astype(bool)
    df_edges.loc[ed_ix2label,"type"] = 'interform-fault'
    df_edges.loc[ed_ix2label[ix_sf],"type"] = 'same-fault'
    df_edges.loc[ed_ix2label[ix_ff],"type"] = 'fault-fault'
    df_edges.loc[ed_ix2label[ix_si],"type"] = 'same-interform'
    df_edges.loc[ed_ix2label[ix_ii],"type"] = 'interform-interform'

    # REMOVE INTERFORM-INTERFORM EDGES THAT DO NOT SHARE A COMMON GEOLOGICAL CODE
    df_edges.reset_index(drop=True, inplace=True)
    ix_interforminterform = np.asarray(np.where(df_edges['type']=='interform-interform')).flatten()
    df_interforminterform = df_edges.loc[ix_interforminterform,['id_node_src','id_node_tgt']].copy()
    ix_node_src = np.asarray(df_interforminterform['id_node_src'].values.astype(int)).flatten()   
    tmp_src = np.asarray(df_nodes.loc[ix_node_src,['interform_geocode_a','interform_geocode_b']].values).astype(int)
    ix_node_tgt = np.asarray(df_interforminterform['id_node_tgt'].values.astype(int)).flatten()
    tmp_tgt = np.asarray(df_nodes.loc[ix_node_tgt,['interform_geocode_a','interform_geocode_b']].values).astype(int)
    no_common_geol = ((tmp_src[:,0]-tmp_tgt[:,0])*(tmp_src[:,0]-tmp_tgt[:,1])*(tmp_src[:,1]-tmp_tgt[:,0])*(tmp_src[:,1]-tmp_tgt[:,1]))!=0
    ix2drop = np.where(no_common_geol==True)
    df_edges.drop(ix_interforminterform[ix2drop],inplace=True)
    df_edges.reset_index(drop=True, inplace=True)
    
    # duplicate edges for non-oriented graph
    if verb: print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - BIDIRECTIONAL edges') 
    df_tmpEdges = pd.DataFrame({'id_node_src':df_edges['id_node_tgt'].values,
                                'id_node_tgt':df_edges['id_node_src'].values,
                                'type':df_edges['type'].values,
                                'label':df_edges['label'].values})
    df_edges = pd.concat([df_edges,df_tmpEdges])
    del df_tmpEdges
    df_edges.drop_duplicates(subset=['id_node_src','id_node_tgt'], keep='first', inplace=True)
    df_edges.reset_index(drop=True, inplace=True)
    
    ix_lbl = np.asarray(np.where(df_edges["label"].isna())).flatten()
    lab1 = df_nodes.loc[df_edges.iloc[ix_lbl,0].values,"geocode"].values #.astype(str)
    lab2 = df_nodes.loc[df_edges.iloc[ix_lbl,1].values,"geocode"].values #.astype(str)
    strlabel = lab1 + ' - ' + lab2
    df_edges.loc[ix_lbl,"label"] = strlabel


    # Add orientation and distance from source to target
    if verb: print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - ADDING ORIENTATION AND LENGTH') 
    df_edges['vx'] = (df_nodes.loc[df_edges['id_node_tgt'].values,'X'].values 
                      - df_nodes.loc[df_edges['id_node_src'].values,'X'].values)
    df_edges['vy'] = (df_nodes.loc[df_edges['id_node_tgt'].values,'Y'].values 
                      - df_nodes.loc[df_edges['id_node_src'].values,'Y'].values)
    df_edges['vz'] = (df_nodes.loc[df_edges['id_node_tgt'].values,'Z'].values 
                      - df_nodes.loc[df_edges['id_node_src'].values,'Z'].values)
    df_edges['length'] = np.sqrt((df_edges['vx'].values)**2 +
                                  (df_edges['vy'].values)**2 +
                                  (df_edges['vz'].values)**2 )

    if simplify==True:
        # Simplify geol feature connections
        if verb: print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - SIMPLIFY FAULT-FAULT edges') 
        # fault-fault
        df_edges.reset_index(drop=True, inplace=True)
        ixtmp  = np.asarray(np.where(df_edges['type']=='fault-fault')).flatten()
        df_tmp = df_edges.loc[ixtmp,:].groupby(['label','id_node_src']).agg({'length': ['min', 'count']})
        df_tmp.reset_index(level=-1, inplace=True)
        ix2keep = np.asarray(np.where(df_tmp.iloc[:,-1]>1)).flatten()
        if len(ix2keep)>0:
            tmp_array = df_tmp.iloc[ix2keep,:].values
            ix2drop=np.array([]).astype(int)
            for i in range(len(ix2keep)):
                ix2droptmp = np.asarray(np.where((df_edges.loc[ixtmp,'id_node_src']==tmp_array[i,0]) & (df_edges.loc[ixtmp,'length']>tmp_array[i,1]))).flatten()
                ix2drop = np.concatenate((ix2drop,ix2droptmp))
            df_edges.drop(ixtmp[ix2drop],inplace=True)
        del df_tmp,ixtmp
        df_edges.reset_index(drop=True, inplace=True)
        ixtmp  = np.asarray(np.where(df_edges['type']=='fault-fault')).flatten()
        df_tmp = df_edges.loc[ixtmp,:].groupby(['label','id_node_tgt']).agg({'length': ['min', 'count']})
        df_tmp.reset_index(level=-1, inplace=True)
        ix2keep = np.asarray(np.where(df_tmp.iloc[:,-1]>1)).flatten()
        if len(ix2keep)>0:
            tmp_array = df_tmp.iloc[ix2keep,:].values
            ix2drop=np.array([]).astype(int)
            for i in range(len(ix2keep)):
                ix2droptmp = np.asarray(np.where((df_edges.loc[ixtmp,'id_node_tgt']==tmp_array[i,0]) & (df_edges.loc[ixtmp,'length']>tmp_array[i,1]))).flatten()
                ix2drop = np.concatenate((ix2drop,ix2droptmp))
            df_edges.drop(ixtmp[ix2drop],inplace=True)
        del df_tmp,ixtmp
        
        # Simplify geol feature connections
        if verb: print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - SIMPLIFY INTERFORM-FAULT edges') 
        # fault-inter 'interform-fault'
        df_edges.reset_index(drop=True, inplace=True)
        ixtmp  = np.asarray(np.where(df_edges['type']=='interform-fault')).flatten()
        df_tmp = df_edges.loc[ixtmp,:].groupby(['label','id_node_src']).agg({'length': ['min', 'count']})
        df_tmp.reset_index(level=-1, inplace=True)
        ix2keep = np.asarray(np.where(df_tmp.iloc[:,-1]>1)).flatten()
        if len(ix2keep)>0:
            tmp_array = df_tmp.iloc[ix2keep,:].values
            ix2drop=np.array([]).astype(int)
            for i in range(len(ix2keep)):
                ix2droptmp = np.asarray(np.where((df_edges.loc[ixtmp,'id_node_src']==tmp_array[i,0]) & (df_edges.loc[ixtmp,'length']>tmp_array[i,1]))).flatten()
                ix2drop = np.concatenate((ix2drop,ix2droptmp))
            df_edges.drop(ixtmp[ix2drop],inplace=True)
        del df_tmp,ixtmp
        df_edges.reset_index(drop=True, inplace=True)
        ixtmp  = np.asarray(np.where(df_edges['type']=='interform-fault')).flatten()
        df_tmp = df_edges.loc[ixtmp,:].groupby(['label','id_node_tgt']).agg({'length': ['min', 'count']})
        df_tmp.reset_index(level=-1, inplace=True)
        ix2keep = np.asarray(np.where(df_tmp.iloc[:,-1]>1)).flatten()
        if len(ix2keep)>0:
            tmp_array = df_tmp.iloc[ix2keep,:].values
            ix2drop=np.array([]).astype(int)
            for i in range(len(ix2keep)):
                ix2droptmp = np.asarray(np.where((df_edges.loc[ixtmp,'id_node_tgt']==tmp_array[i,0]) & (df_edges.loc[ixtmp,'length']>tmp_array[i,1]))).flatten()
                ix2drop = np.concatenate((ix2drop,ix2droptmp))
            df_edges.drop(ixtmp[ix2drop],inplace=True)
        del df_tmp,ixtmp
    
        
        # Simplify geol feature connections
        if verb: print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - SIMPLIFY INTERFORM-INTERFORM edges') 
        # inter-inter 'interform-interform'
        df_edges.reset_index(drop=True, inplace=True)
        ixtmp  = np.asarray(np.where(df_edges['type']=='interform-interform')).flatten()
        df_tmp = df_edges.loc[ixtmp,:].groupby(['label','id_node_src']).agg({'length': ['min', 'count']})
        df_tmp.reset_index(level=-1, inplace=True)
        ix2keep = np.asarray(np.where(df_tmp.iloc[:,-1]>1)).flatten()
        if len(ix2keep)>0:
            tmp_array = df_tmp.iloc[ix2keep,:].values
            ix2drop=np.array([]).astype(int)
            for i in range(len(ix2keep)):
                ix2droptmp = np.asarray(np.where((df_edges.loc[ixtmp,'id_node_src']==tmp_array[i,0]) & (df_edges.loc[ixtmp,'length']>tmp_array[i,1]))).flatten()
                ix2drop = np.concatenate((ix2drop,ix2droptmp))
            df_edges.drop(ixtmp[ix2drop],inplace=True)
        del df_tmp,ixtmp
        df_edges.reset_index(drop=True, inplace=True)
        ixtmp  = np.asarray(np.where(df_edges['type']=='interform-interform')).flatten()
        df_tmp = df_edges.loc[ixtmp,:].groupby(['label','id_node_tgt']).agg({'length': ['min', 'count']})
        df_tmp.reset_index(level=-1, inplace=True)
        ix2keep = np.asarray(np.where(df_tmp.iloc[:,-1]>1)).flatten()
        if len(ix2keep)>0:
            tmp_array = df_tmp.iloc[ix2keep,:].values
            ix2drop=np.array([]).astype(int)
            for i in range(len(ix2keep)):
                ix2droptmp = np.asarray(np.where((df_edges.loc[ixtmp,'id_node_tgt']==tmp_array[i,0]) & (df_edges.loc[ixtmp,'length']>tmp_array[i,1]))).flatten()
                ix2drop = np.concatenate((ix2drop,ix2droptmp))
            df_edges.drop(ixtmp[ix2drop],inplace=True)
        del df_tmp,ixtmp
        # END OF SIMPLIFICATION
  
    # remove columns from node dataframe
    df_nodes.drop(axis=1,columns=['interform_geocode_a','interform_geocode_b'],inplace=True)

    
    # reset edges index
    df_edges.sort_values(by=['id_node_src','id_node_tgt'], axis=0, ascending=True, inplace=True)
    df_edges.reset_index(drop=True, inplace=True)

    # REMOVE EDGES CROSSING FAULTS 
    if verb: print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - REMOVE EDGES CROSSING FAULTS')
    x = np.sort(np.unique(nd_X))
    y = np.sort(np.unique(nd_Y))
    z = np.sort(np.unique(nd_Z))
    # order x y z along dimensions
    if len(x)==dim[0]:
        v0 = x
        pts0 = np.asarray(df_nodes['X']).astype(float)
    elif len(y)==dim[0]:
        v0 = y
        pts0 = np.asarray(df_nodes['Y']).astype(float)
    elif len(z)==dim[0]:
        v0 = z
        pts0 = np.asarray(df_nodes['Z']).astype(float)
    if len(y)==dim[1]:
        v1 = y
        pts1 = np.asarray(df_nodes['Y']).astype(float)
    elif len(x)==dim[1]:
        v1 = x
        pts1 = np.asarray(df_nodes['X']).astype(float)
    elif len(z)==dim[1]:
        v1 = z
        pts1 = np.asarray(df_nodes['Z']).astype(float)
    if len(z)==dim[2]:
        v2 = z
        pts2 = np.asarray(df_nodes['Z']).astype(float)
    elif len(x)==dim[2]:
        v2 = x
        pts2 = np.asarray(df_nodes['X']).astype(float)
    elif len(y)==dim[2]:
        v2 = y
        pts2 = np.asarray(df_nodes['Y']).astype(float)
    pts = np.concatenate((np.reshape(pts0,(len(df_nodes),1)), np.reshape(pts1,(len(df_nodes),1)), np.reshape(pts2,(len(df_nodes),1))), axis=1)
    fault_topo_nodes = np.zeros((len(df_nodes),nbfaults)).astype(int)
    for f in range(nbfaults):
        if verb: print('fault_names[fault_id]:',fault_names[f])
        cur_topo_fault = nd_topo_faults[:,:,:,f]
        # cur_ix = np.where(cur_topo_fault!=0)
        if verb: print('np.unique(cur_topo_fault):',np.unique(cur_topo_fault))
        cur_interp = RegularGridInterpolator((v0,v1,v2),cur_topo_fault,method='nearest')
        fault_topo_nodes[:,f] = cur_interp(pts).astype(int)
    crossing_fault = np.sum( 1*( (fault_topo_nodes[df_edges['id_node_src'].values.astype(int),:] 
                                 *fault_topo_nodes[df_edges['id_node_tgt'].values.astype(int),:])
                                ==-1)
                            , axis=1)
    ix2drop = np.asarray(np.where( (crossing_fault>0) &
                                   (~df_edges['type'].isin(['fault-fault','fault-formation','interform-fault','same-fault']))
                                  )).flatten()
    if verb: 
        print('Summary of edges crossing faults to remove:')
        print(df_edges.loc[ix2drop,].groupby(['type']).size())
    df_edges.drop(ix2drop,inplace=True)
    df_edges.reset_index(drop=True, inplace=True)
    if verb: print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - EDGES CROSSING FAULTS REMOVED.')
    
    if verb:
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE - END')
        print('df_nodes: ' + str(sys.getsizeof(df_nodes)/mem_factor) + ' ' + mem_unit +
              ' - '+str(len(df_nodes))+' nodes')
        print(df_nodes.groupby(['description']).size())
        print('df_edges: ' + str(sys.getsizeof(df_edges)/mem_factor) + ' ' + mem_unit +
              ' - '+str(len(df_edges))+' edges')
        print(df_edges.groupby(['type']).size())       

    return df_nodes,df_edges

# Memory profiling https://github.com/spyder-ide/spyder-memory-profiler 
# Add a @profile decorator to the functions that you wish to profile then Ctrl+Shift+F10 to run the profiler on the current script, or go to Run > Profile memory line by line.
# @profile
def reggrid2nxGraph(nd_X,nd_Y,nd_Z,nd_lithocodes,nd_topo_faults,fault_names,destination="",unique_edges=True,simplify=True,verb=False,csvxpt=False,edgeGeocode=True):
    """"Conversion of a regular grid voxet into a graph

    Parameters
    ----------
    nd_X : numpy.ndarray of floats 
        flattened array of the regular grid x-coordinates voxet
    nd_Y : numpy.ndarray of floats
        flattened array of the regular grid y-coordinates voxet
    nd_Z : numpy.ndarray of floats
        flattened array of the regular grid z-coordinates voxet
    nd_lithocodes : numpy.ndarray of integers
        flattened array of the regular grid lithocodes voxet
    nd_topo_faults : numpy.ndarray of integers
        2D array of the fault topology matrices of shape (len(nd_X),nfaults), with nfaults as the number of faults
    fault_names : list of strings
        fault names list of length nfaults
    destination : 
        destination folder to save 'model-graph.gml'
    unique_edges : bool
        optional boolean True (default) or False - remove duplicate (src-tgt) edges
    simplify : bool
        optional boolean True (default) or False - simplify edges when several edges connect one node to several other of the same entity, keeps the one with the shortest distance
    verb : bool
        optional boolean True or False (default) - verbose - print what it is doing 
    csvxpt : bool
        optional boolean True or False (default) - export node and edges dataframe to .csv files in the destination folder   
    edgeGeocode bool
        optional boolean True (default) or False - Adds the source and target geocodes to the df_edges dataframe

    Returns
    -------
    G : DiGraph
        networkx graph
    df_nodes : DataFrame
        pandas DataFrame listing the graph nodes
    df_edges : DataFrame
        pandas DataFrame listing the graph edges
    """
    df_nodes,df_edges = reggrid_topology_graph(nd_X,nd_Y,nd_Z,nd_lithocodes,nd_topo_faults,fault_names,unique_edges=unique_edges,simplify=simplify,verb=verb)
    # add geocodes
    if edgeGeocode: add_edges_geocodes(df_edges,df_nodes)
    if verb==True:
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - BUILDING GRAPH')
    G = nx.from_pandas_edgelist(df_edges, source='id_node_src', target='id_node_tgt', edge_attr=True,create_using=nx.DiGraph())    #,create_using=nx.DiGraph()
    G.is_directed()
    edgesid = np.unique(df_edges[['id_node_src','id_node_tgt']].values.astype(int).flatten())
    edgeless_nodes = np.setdiff1d(df_nodes['id'].values, edgesid)
    df_nodes2 = df_nodes.copy()
    df_nodes2.drop(edgeless_nodes,inplace=True)
    # Iterate over df rows and set the source and target nodes' attributes for each row:
    for index, row in df_nodes2.iterrows():
        # print('index:',str(index),' - row:',str(row))
        G.nodes[row['id']]['X'] = row['X']
        G.nodes[row['id']]['Y'] = row['Y']
        G.nodes[row['id']]['Z'] = row['Z']
        G.nodes[row['id']]['geocode'] = row['geocode']
        G.nodes[row['id']]['description'] = row['description']
        G.nodes[row['id']]['orthodim'] = row['orthodim']
    # Write outputs
    if ((isinstance(destination, str)) & (destination!="")):
        if os.path.isdir(destination)==False:
            os.mkdir(destination)
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - EXPORTING GML GRAPH')
        nx.write_gml(G, destination+"/model-graph.gml")
        if (verb==True & csvxpt==True) :
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - EXPORTING CSV NODES')
            df_nodes.to_csv(destination+"/model-nodes.csv",index=False)
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - EXPORTING CSV EDGES')
            df_edges.to_csv(destination+"/model-edges.csv",index=False)
    if verb==True:
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - reggrid2nxGraph END')
    return G,df_nodes,df_edges,edgeless_nodes

def add_new_edges(tmpID1,tmpID2,df_edges,verb=False,suffix=""):
    tmp_ix = np.asarray(np.where((tmpID1*tmpID2)>0)).flatten()
    if len(tmp_ix)>0:
        df_tmpEdges = pd.DataFrame({'id_node_src':tmpID1[tmp_ix],'id_node_tgt':tmpID2[tmp_ix]})
        if verb==True:
            print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - MEMORY USAGE GLOBAL - '+suffix)
            print('df_tmpEdges: ' + str(sys.getsizeof(df_tmpEdges)/mem_factor) + ' ' + mem_unit)
        df_edges = pd.concat([df_edges,df_tmpEdges])
        del df_tmpEdges
    return df_edges
        
def extract_shift_dim(input_mx,axis_mx):
    tmpdim = input_mx.shape
    out1 = input_mx.take(indices=range(1, tmpdim[axis_mx]), axis=axis_mx).flatten()
    out2 = input_mx.take(indices=range(0, tmpdim[axis_mx]-1), axis=axis_mx).flatten()
    return out1,out2

def add_edges_geocodes(df_edges,df_nodes):
    """
    Parameters
    ----------
    df_edges : DataFrame
        pandas DataFrame listing the graph edges
    df_nodes : DataFrame
        pandas DataFrame listing the graph nodes

    Returns
    -------
    Adds the source and target geocodes to the df_edges dataframe
    in two columns: 'geocode_src' and 'geocode_tgt' 

    """
    df_edges['geocode_src']=df_nodes.loc[df_edges['id_node_src'].values.astype(int),'geocode'].values    
    df_edges['geocode_tgt']=df_nodes.loc[df_edges['id_node_tgt'].values.astype(int),'geocode'].values
    return

