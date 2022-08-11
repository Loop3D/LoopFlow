from LoopFlow import topological_analysis as ta
from LoopStructural.analysis._topology import calculate_fault_topology_matrix
import numpy as np
import pandas as pd
import networkx as nx
import os
from datetime import datetime
from networkx.algorithms.flow import boykov_kolmogorov


def graph_from_model(model,voxel_size,bbox2,destination):  
    if(not os.path.isdir(destination)):
        os.mkdir(destination)

    maxz=bbox2.iloc[0]['upper']
    minz=bbox2.iloc[0]['lower']

    length_x = bbox2.iloc[0]['maxx'] - bbox2.iloc[0]['minx']
    length_y = bbox2.iloc[0]['maxy'] - bbox2.iloc[0]['miny']
    length_z = maxz - minz

    nnx = int(length_x/voxel_size)
    nny = int(length_y/voxel_size)
    nnz = int(length_z/voxel_size)

    x = np.linspace(bbox2.iloc[0]['minx'],bbox2.iloc[0]['maxx'],nnx)
    y = np.linspace(bbox2.iloc[0]['miny'],bbox2.iloc[0]['maxy'],nny)
    z = np.linspace(minz,maxz,nnz)

    #ext = [bbox2.iloc[0]['miny'],bbox2.iloc[0]['maxy'],bbox2.iloc[0]['maxx'],bbox2.iloc[0]['minx']] # left right bottom top

    xxx, yyy, zzz = np.meshgrid(x, y, z, indexing='ij') # build mesh
    xyz = np.array([xxx.flatten(), yyy.flatten(), zzz.flatten()]).T # build array for LS lithocode evaluation function
    xyzLitho = model.evaluate_model(xyz,scale=True)
    nd_lithocodes = np.reshape(xyzLitho,(nnx,nny,nnz))
    print('nnx,nny,nnz',nnx,nny,nnz)
    nd_X = np.reshape(xyz[:,0],(nnx,nny,nnz))
    nd_Y = np.reshape(xyz[:,1],(nnx,nny,nnz))
    nd_Z = np.reshape(xyz[:,2],(nnx,nny,nnz))

    fault_topo_mat = calculate_fault_topology_matrix(model,xyz,scale=True)
    import pickle
    save_pickle=True
    if(save_pickle):
        with open(destination+'/xyzLitho.pickle', 'wb') as f:
            pickle.dump(xyzLitho, f)
        with open(destination+'/xyz.pickle', 'wb') as f:
            pickle.dump(xyz, f)
        with open(destination+'/fault_topo_mat.pickle', 'wb') as f:
            pickle.dump(fault_topo_mat, f)

    nbfaults = fault_topo_mat.shape[-1]
    nd_topo_faults = np.reshape(fault_topo_mat,(nnx,nny,nnz,nbfaults))

    # plt.imshow(nd_topo_faults[:,:,-1,5],extent=ext),plt.xlabel('y'),plt.ylabel('x'),plt.title('topo fault 0'),plt.show
    fault_names = []
    for f in range(nbfaults):
        fault_names.append("fault"+str(f)) 
    
    Graw,df_nodes_raw,df_edges_raw = ta.reggrid2nxGraph(nd_X,nd_Y,nd_Z,nd_lithocodes,nd_topo_faults,fault_names,destination,unique_edges=True,simplify=False,verb=False,csvxpt=True) #,destination
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - GRAPH CALCULATED')

    return(Graw,df_nodes_raw,df_edges_raw)

def assign_weights(Graw,scenario,source,target,fast_litho,faults_only,bbox2,px,py,pz,range):

    maxz=bbox2.iloc[0]['upper']
    minz=bbox2.iloc[0]['lower']
    maxx=bbox2.iloc[0]['maxx']
    minx=bbox2.iloc[0]['minx']
    maxy=bbox2.iloc[0]['maxy']
    miny=bbox2.iloc[0]['miny']

    if(scenario=='fast_both'):
        fault_node=1.0
        geological_formation_slow=5.0
        geological_formation_fast=5.0
        interformation_node=5.0

        fault_formation=1.0
        same_fault=0.1
        fault_fault=1.0
        interform_fault=1.0
        interform_formation=1.0
        interform_interform=1.0
        same_interform=5.0

        fast_formation_code=fast_litho
    elif(scenario=='fast_strat_contacts'): # and slow faults
        fault_node=5.0
        geological_formation_slow=1.0
        geological_formation_fast=1.0
        interformation_node=1.0

        fault_formation=1.0
        same_fault=5.0
        fault_fault=5.0
        interform_fault=1.0
        interform_formation=1.0
        interform_interform=1.0
        same_interform=5.0

        fast_formation_code=fast_litho
    elif(scenario=='fast_faults'): # and slow strat
        fault_node=1.0
        geological_formation_slow=5.0
        geological_formation_fast=1.0
        interformation_node=5.0

        fault_formation=1.0
        same_fault=0.1
        fault_fault=1.0
        interform_fault=5.0
        interform_formation=5.0
        interform_interform=5.0
        same_interform=5.0

        fast_formation_code=fast_litho
    elif(scenario=='fault_barriers_not_paths'): # and fast strat
        fault_node=1000.0
        geological_formation_slow=200.0
        geological_formation_fast=1.0
        interformation_node=1.0

        fault_formation=1000.0
        same_fault=1000.0
        fault_fault=1000.0
        interform_fault=1000.0
        interform_formation=25.0
        interform_interform=25.0
        same_interform=25.0

        fast_formation_code=fast_litho
    elif(scenario=='fault_barriers_but_paths_and_fast_strat'):
        fault_node=5.0
        geological_formation_slow=1.0
        geological_formation_fast=1.0
        interformation_node=1.0

        fault_formation=5.0
        same_fault=1.0
        fault_fault=1.0
        interform_fault=5.0
        interform_formation=1.0
        interform_interform=1.0
        same_interform=5.0

        fast_formation_code=fast_litho
    elif(scenario=="homogeneous"): # homogeneous
        fault_node=1.0
        geological_formation_slow=1.0
        geological_formation_fast=1.0
        interformation_node=1.0

        fault_formation=1.0
        same_fault=1.0
        fault_fault=1.0
        interform_fault=1.0
        interform_formation=1.0
        interform_interform=1.0
        same_interform=1.0

        fast_formation_code=[-99]
    else: # custom
        fault_node=scenario['fault_node']
        geological_formation_slow=scenario['geological_formation_slow']
        geological_formation_fast=scenario['geological_formation_fast']
        interformation_node=scenario['interformation_node']

        interform_formation=scenario['interform_formation']
        fault_formation=scenario['fault_formation']
        same_fault=scenario['same_fault']
        interform_fault=scenario['interform_fault']
        fault_fault=scenario['fault_fault']
        same_interform=scenario['same_interform']
        interform_interform=scenario['interform_interform']

        fast_formation_code=scenario['fast_formation_code']
        scenario='custom'
       
        
        
    length_scale_max=range*0.866 # half width * (3^0.5)/2 
    G=Graw.to_undirected()
    for n in G.nodes:
        if(G.nodes[n]['description']=='fault node'):
            G.nodes[n]['weight']=fault_node
        elif(G.nodes[n]['description']=='geological formation'):
            if(G.nodes[n]['geocode']==fast_formation_code):
                G.nodes[n]['weight']=geological_formation_fast
            else:
                G.nodes[n]['weight']=geological_formation_slow
        elif(G.nodes[n]['description']=='interformation node'):
            G.nodes[n]['weight']=5.0
    for e in G.edges:
        scale=G.edges[e]['length']/length_scale_max
        #scale=1
        if(G.edges[e]['type']=='fault-formation'):
            G.edges[e]['weight']=fault_formation*scale
            G.edges[e]['capacity']=1/fault_formation
        elif(G.edges[e]['type']=='same-fault'):
            G.edges[e]['weight']=same_fault*scale
            G.edges[e]['capacity']=1/same_fault
        elif(G.edges[e]['type']=='fault-fault'):
            G.edges[e]['weight']=fault_fault*scale
            G.edges[e]['capacity']=1/fault_fault
        elif(G.edges[e]['type']=='interform-fault'):
            G.edges[e]['weight']=interform_fault*scale
            G.edges[e]['capacity']=1/interform_fault
        elif(G.edges[e]['type']=='interform-formation'):
            G.edges[e]['weight']=interform_formation*scale
            G.edges[e]['capacity']=1/interform_formation
        elif(G.edges[e]['type']=='intra-formation'):
            if(G.edges[e]['label'].replace('geol_','') in fast_formation_code):
                G.edges[e]['weight']=geological_formation_fast*scale
                G.edges[e]['capacity']=1/geological_formation_fast
            else:
                G.edges[e]['weight']=geological_formation_slow*scale
                G.edges[e]['capacity']=1/geological_formation_slow
        elif(G.edges[e]['type']=='interform-interform'):
            G.edges[e]['weight']=interform_interform*scale
            G.edges[e]['capacity']=1/interform_interform
        elif(G.edges[e]['type']=='same-interform'):
            G.edges[e]['weight']=same_interform*scale
            G.edges[e]['capacity']=1/same_interform


    if(source=='deep_line'):
        G=add_deep_line_node(G,-1,minx,maxx,minz,range*2,faults_only)
    elif(source=='point'):
        add_point_node(G,-1,px,py,pz,range*2,faults_only)
    else:
        G=add_side_node(G,-1,bbox2,range*2,source,faults_only)

   
    if(target=='deep_line'):
        G=add_deep_line_node(G,-2,minx,maxx,minz,range*2,faults_only)
    elif(target=='point'):
        add_point_node(G,-2,px,py,pz,range*2,faults_only)
    elif(target=='none'):
        pass
    else:
        G=add_side_node(G,-2,bbox2,range*2,target,faults_only)
  
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - WEIGHTS ASSIGNED')
    return(G,scenario)

def add_deep_line_node(G,nodeid,minx,maxx,minz,range,faults_only):

    range_min=minx+((maxx-minx)/2)-range
    range_max=maxx+((maxx-minx)/2)+range
    #print("range=",range)
    #print("range_min=",range_min)
    #print("range_max=",range_max)

    G.add_node(nodeid)
    if(nodeid==-1):
        name='node source'
    else:
        name='node target'
    G.nodes[nodeid]['label']=name
    G.nodes[nodeid]['X']= 0
    G.nodes[nodeid]['Y']= 0
    G.nodes[nodeid]['Z']= 0
    G.nodes[nodeid]['geocode']= name
    G.nodes[nodeid]['description']= name
    G.nodes[nodeid]['orthodim']= 0
    G.nodes[nodeid]['weight']= 1.0

    for n in G.nodes():
        if(n >=0):
            if(G.nodes[n]['Z']<minz+100 and G.nodes[n]['X']<range_max and G.nodes[n]['X']>range_min  ):
                if((faults_only and G.nodes[n]['description']=='fault node') or not faults_only ):
                    G=join_node(G,n,name,'deep')
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - LINE SOURCE ADDED')

    return(G)

def join_node(G,n,name,type):
    if(name=='node target'):
        nodeid=-2
        G.add_edge(n,nodeid)
        G.edges[n,nodeid]['type']= type+" source"
        G.edges[n,nodeid]['label']= type+" source"
        G.edges[n,nodeid]['geocode_src']= 0
        G.edges[n,nodeid]['geocode_tgt']= 0
        G.edges[n,nodeid]['vx']= 0.0
        G.edges[n,nodeid]['vy']= 0.0
        G.edges[n,nodeid]['vz']= 0.0
        G.edges[n,nodeid]['length']= 0
        G.edges[n,nodeid]['weight']= 0.1
        G.edges[n,nodeid]['capacity']= 100000
    else:
        nodeid=-1
        G.add_edge(nodeid,n)
        G.edges[nodeid,n]['type']= type+" source"
        G.edges[nodeid,n]['label']= type+" source"
        G.edges[nodeid,n]['geocode_src']= 0
        G.edges[nodeid,n]['geocode_tgt']= 0
        G.edges[nodeid,n]['vx']= 0.0
        G.edges[nodeid,n]['vy']= 0.0
        G.edges[nodeid,n]['vz']= 0.0
        G.edges[nodeid,n]['length']= 0
        G.edges[nodeid,n]['weight']= 0.1
        G.edges[nodeid,n]['capacity']= 100000
    return(G)

def add_point_node(G,nodeid,x,y,z,range,faults_only):

    pt_xmin=x-range
    pt_xmax=x+range
    pt_ymin=y-range
    pt_ymax=y+range
    pt_zmin=z-range
    pt_zmax=z+range


    G.add_node(nodeid)
    if(nodeid==-1):
        name='node source'
    else:
        name='node target'
    G.nodes[nodeid]['label']=name
    G.nodes[nodeid]['X']= 0
    G.nodes[nodeid]['Y']= 0
    G.nodes[nodeid]['Z']= 0
    G.nodes[nodeid]['geocode']= name
    G.nodes[nodeid]['description']= name
    G.nodes[nodeid]['orthodim']= 0
    G.nodes[nodeid]['weight']= 1.0

    for n in G.nodes():
        if(n >=0):
            if(G.nodes[n]['X']<pt_xmax and G.nodes[n]['X']>pt_xmin and 
                G.nodes[n]['Y']<pt_ymax and G.nodes[n]['Y']>pt_ymin and
                G.nodes[n]['Z']<pt_zmax and G.nodes[n]['Z']>pt_zmin):
                    if((faults_only and G.nodes[n]['description']=='fault node') or not faults_only ):
                        G=join_node(G,n,name,'point')

    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - POINT SOURCE ADDED')
    return(G)

def add_side_node(G,nodeid,bbox2,range,side,faults_only):
    maxz=bbox2.iloc[0]['upper']
    minz=bbox2.iloc[0]['lower']
    maxx=bbox2.iloc[0]['maxx']
    minx=bbox2.iloc[0]['minx']
    maxy=bbox2.iloc[0]['maxy']
    miny=bbox2.iloc[0]['miny']

    if(side=='west'):
        range_min=minx
        range_max=minx+range
    elif(side=='east'):
        range_min=maxx-range
        range_max=maxx
    elif(side=='north'):
        range_min=maxy-range
        range_max=maxy
    elif(side=='south'):
        range_min=miny
        range_max=miny+range
    elif(side=='top'):
        range_min=maxz-range
        range_max=maxz
    else: #(side=='base')
        range_min=minz
        range_max=minz+range


    #print("range=",range,minx,maxx)
    #print("range_min=",range_min)
    #print("range_max=",range_max)

    G.add_node(nodeid)
    if(nodeid==-1):
        name='node source'
    else:
        name='node target'
    G.nodes[nodeid]['label']=name
    G.nodes[nodeid]['X']= 0.0
    G.nodes[nodeid]['Y']= 0.0
    G.nodes[nodeid]['Z']= 0.0
    G.nodes[nodeid]['geocode']= name
    G.nodes[nodeid]['description']= name
    G.nodes[nodeid]['orthodim']= 0
    G.nodes[nodeid]['weight']= 1.0

    for n in G.nodes():
        if(n>=0):
            in_range=False
            if(side=='west' or side=='east' ):
                if( G.nodes[n]['X']<=range_max and G.nodes[n]['X']>=range_min  ): 
                    in_range=True
            elif(side=='north' or side=='south' ):
                if( G.nodes[n]['Y']<=range_max and G.nodes[n]['Y']>=range_min  ): 
                    in_range=True
            elif(side=='top' or side=='base' ):
                if( G.nodes[n]['Z']<=range_max and G.nodes[n]['Z']>=range_min  ): 
                    in_range=True

            if(in_range):
                if((faults_only and G.nodes[n]['description']=='fault node') or not faults_only ):
                    G=join_node(G,n,name,'side')
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - '+side+' NODE ADDED')
    return(G)

    
def calculate_dist(G,df_nodes,voxel_size,bbox2,scenario,destination):
    dx=dy=dz=voxel_size
    minz=bbox2.iloc[0]['lower']
               
    nx.write_gml(G,destination+'/'+scenario+'_model.gml') 
    def_src=pd.DataFrame(index=[-1],data={'id':-1,'X':0,'Y':0,'Z':minz-1,'geocode':'deep source','description':'deep source','orthodim':0})
    df_nodes=pd.concat([df_nodes,def_src])


    def func(u, v, d):
        node_u_wt = G.nodes[u].get("weight", 1)
        node_v_wt = G.nodes[v].get("weight", 1)
        edge_wt = d.get("weight", 1)
        return edge_wt
        #return node_u_wt / 2 + node_v_wt / 2 + edge_wt
        
    X=0
    Y=0
    Z=minz-1

       
    #try:
    if(True):
        close=0.51
        source_node=df_nodes.copy()
        source_node=source_node[(source_node['description']=='geological formation')|(source_node['description']=='deep source')]
        source_node=source_node[(source_node['X']>X-(dx*close)) & (source_node['X']<X+(dx*close))]
        source_node=source_node[(source_node['Y']>Y-(dy*close)) & (source_node['Y']<Y+(dy*close))]
        source_node=source_node[(source_node['Z']>Z-(dz*close)) & (source_node['Z']<Z+(dz*close))]
        source=source_node.iloc[0]['id']
        print('source=',source)
        distance, path=nx.single_source_dijkstra(G, source, target=None, cutoff=None, weight='weight')

        voxet={}

        for n in G.nodes:
            if(n!=-2):
                voxet[n]={'dist':distance[n],'weight':G.nodes[n]['weight'],'X':G.nodes[n]['X'],'Y':G.nodes[n]['Y'],'Z':G.nodes[n]['Z']}

        voxet_df=pd.DataFrame.from_dict(voxet,orient='index')
        voxet_df[:-2].to_csv(destination+'/'+scenario+'_dist_voxet.csv')
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - DISTS CALCULATED')

        return(voxet_df,distance,path)


                    
def calculate_paths(path,df_nodes,scenario,destination):
    from collections import Counter
    flat_list=[val for vals in path.values() for val in vals]
    node_count=Counter(flat_list)
    counts=pd.DataFrame.from_dict(node_count,orient='index').sort_index()
    df_nodes['count']=counts[0]
    df_nodes[:-1].to_csv(destination+'/'+scenario+'_path_count.csv',index=False)
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - PATHS CALCULATED')


def calculate_scenery(G,model,df_nodes,path,scenario,destination):

    # create dict of lithos for strat nodes and empty rows for faults
    node_lith={}

    for i in G.nodes():
        #print(i,G.nodes[i])
        if(G.nodes[i]['description']=='geological formation'):
            node_lith[i]={'type':'litho','label':i,'litho1':G.nodes[i]['geocode'],'litho2':G.nodes[i]['geocode']}
        elif(G.nodes[i]['description']=='interformation node'):
            geocode1=int(G.nodes[i]['geocode'].replace("_","-").split("-")[2])
            geocode2=int(G.nodes[i]['geocode'].replace("_","-").split("-")[4])
            node_lith[i]={'type':'litho','label':i,'litho1':geocode1,'litho2':geocode2}
        else:
            node_lith[i]={'type':'fault','label':-1,'litho1':-1,'litho2':-1}
            

    flist_edges=nx.to_pandas_edgelist(G)
    flist_edges_f=flist_edges[(flist_edges['type']=='fault-formation')|(flist_edges['type']=='interform-fault')]

    for ind,fl in flist_edges_f.iterrows():
        if('Fault_' in str(G.nodes[fl['source']]['geocode'])):
            nodeid=fl['source']
            other_geocode=G.nodes[fl['target']]['geocode']
        else:
            nodeid=fl['target']
            other_geocode=G.nodes[fl['source']]['geocode']
        split_geocode=str(other_geocode).replace("_","-").split("-")
        if(len(split_geocode)==1):
            
            if(node_lith[nodeid]['litho1']==-1):
                node_lith[nodeid]={'type':'fault','label':int(nodeid),'litho1':other_geocode,'litho2':-1}
            else:
                node_lith[nodeid]={'type':'fault','label':int(nodeid),'litho1':node_lith[nodeid]['litho1'],'litho2':other_geocode}
        else:
            #print(str(other_geocode).replace("_","-"))
            node_lith[nodeid]={'type':'fault','label':int(nodeid),'litho1':int(split_geocode[2]),'litho2':int(split_geocode[4])}

    node_lith_df=pd.DataFrame.from_dict(node_lith,orient='index')
         

    first=True
    index=0

    for sg in model.feature_name_index:
        if( not 'Fault' in sg  and not '_unconformity' in sg):
            a_sg_df=pd.DataFrame.from_dict(model.stratigraphic_column[sg],orient='index') 
            a_sg_df['unit']=a_sg_df.index
            a_sg_df=a_sg_df.set_index('id')
            a_sg_df
            if(first):
                sg_df=a_sg_df.copy()
                first=False
            else:
                sg_df=pd.concat([sg_df,a_sg_df])
    """                
    path_litho=[]
    for index,p in enumerate(path):
        step_litho=[]
        for step in path[p]:
            #print(index,'length',len(path[p]),step)
            if(step!=-1):
                step_litho.append(node_lith_df.loc[step]['litho1'])
                step_litho.append(node_lith_df.loc[step]['litho2'])
        if(index%10000==0):
            print(index,len(path))
            
        count_litho=[]
        for i in range(len(sg_df)):
            count_litho.append(step_litho.count(i))
        path_litho.append((p,count_litho))
    """        
    path_litho=[]
    for index,p in enumerate(path):
        step_litho1=[]
        step_litho2=[]
        step_litho1 = [node_lith_df.loc[step]['litho1'] for step in path[p]] 
        step_litho2 = [node_lith_df.loc[step]['litho2'] for step in path[p]] 
        step_litho=step_litho1+step_litho2   
        
        if(step_litho!= None):
            count_litho=[step_litho.count(i) for i in range(len(sg_df))]
            path_litho.append((p,count_litho))
        if(index%10000==0):
            print(index,len(path))

    columns=['a','b']
    path_litho_temp = pd.DataFrame(path_litho[1:], columns = columns)
    columns=[]
    for ind,strat in sg_df.iterrows():
        columns.append(strat['unit'])

    path_litho_df = pd.DataFrame()
    path_litho_df = pd.DataFrame(path_litho_temp['b'].to_list(), columns=columns)
    path_litho_df.index=path_litho_temp['a']
    df_nodes['lkey']=df_nodes.index
    path_litho_df['rkey']=path_litho_df.index

    scenery=df_nodes.merge(path_litho_df, how='outer',left_on='lkey', right_on='rkey')

    scenery.to_csv(destination+'/'+scenario+'_path_scenery.csv')
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - SCENERY CALCULATED')

    return(scenery)

def save_nodes(df_nodes,scenario,destination):
    df_nodes.to_csv(destination+'/'+scenario+'_test-nodes.csv',index=False)
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - NODES SAVED')

def located_edges(df_nodes,df_edges,scenario,destination):
    df_edges_test=df_edges.copy()
    node_src=df_edges['id_node_src']
    node_tgt=df_edges['id_node_tgt']
    X1=df_nodes.loc[node_src].X
    X2=df_nodes.loc[node_tgt].X
    Y1=df_nodes.loc[node_src].Y
    Y2=df_nodes.loc[node_tgt].Y
    Z1=df_nodes.loc[node_src].Z
    Z2=df_nodes.loc[node_tgt].Z
    df_edges_test['X']=(X1.values+X2.values)/2
    df_edges_test['Y']=(Y1.values+Y2.values)/2
    df_edges_test['Z']=(Z1.values+Z2.values)/2

    return(df_edges_test)


def save_edges(df_nodes,df_edges,scenario,destination):
    
    df_edges_test=located_edges(df_nodes,df_edges,scenario,destination)
    df_edges_test.to_csv(destination+'/'+scenario+'_edges.csv',index=False)

    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - EDGES SAVED')

def save_voxels(model,obj_path_dir,bbox2,voxel_size,vtype): #vtype='asc','raw','both'
    maxz=bbox2.iloc[0]['upper']
    minz=bbox2.iloc[0]['lower']
    maxx=bbox2.iloc[0]['maxx']
    minx=bbox2.iloc[0]['minx']
    maxy=bbox2.iloc[0]['maxy']
    miny=bbox2.iloc[0]['miny']

    sizex=int((maxx-minx)/voxel_size)
    sizey=int((maxy-miny)/voxel_size)
    sizez=int((maxz-minz)/voxel_size)
    print('voxel_size=',voxel_size,', saved in X,Y,Z order 16 bit unisgned, X(height)=',sizex,', Y(#ims)=',sizey,', Z(width)=',sizez)
    print('lower south west corner: west=',minx,', south=',miny,', lower=',minz)
    voxels=model.evaluate_model(model.regular_grid(nsteps=(sizex,sizey,sizez),shuffle=False),scale=False)
    voxels=np.rot90(voxels.reshape((sizex,sizey,sizez)),1,(0,2))
    voxels=voxels.reshape((sizex*sizey*sizez))
    if(vtype != 'asc'):
        voxels.astype('int16').tofile(obj_path_dir+'/voxels_'+str(voxel_size)+'.raw')
    else:
        np.savetxt(obj_path_dir+'/voxels_'+str(voxel_size)+'.asc', voxels)
    print('voxels saved in',obj_path_dir)

    f=open(obj_path_dir+'/voxels_'+str(voxel_size)+'_readme.txt','w')
    f.write('voxel_size={}, saved in X,Y,Z order 16 bit unisgned, X(height)={}, Y(#ims)={}, Z(width)={}\n'.format(voxel_size,sizex,sizey,sizez))
    f.write('lower south west corner: west={}, south={}, lower={}'.format(minx,miny,minz))
    f.close()
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - '+vtype+' VOXELS SAVED')


def merge_outputs(voxet_df,df_nodes,scenery,scenario,destination):
    all_nodes=voxet_df[:-1].merge(df_nodes[:-1],how='outer',left_index=True,right_index=True)
    all_nodes=all_nodes.merge(scenery,how='outer',left_index=True,right_index=True)

    drop_candidates=['X_x', 'Y_x', 'Z_x', 'id_x', 'X_y', 
           'Y_y', 'Z_y', 'geocode_y', 'description_y',
           'orthodim_y', 'id_y','lkey', 'rkey','lkey_x', 'rkey_y','count_y']
    
    for d in drop_candidates:
        if d in all_nodes.columns:
            all_nodes=all_nodes.drop([d],axis=1)
    if 'count_x' in all_nodes.columns:
            all_nodes=all_nodes.rename(columns={'count_x':'count'})

    all_nodes=all_nodes.rename(columns={'geocode_x':'geocode','description_x':'description', 'orthodim_x':'orthodim'})
    all_nodes['dist_inv']=1/all_nodes.dist.pow(0.5)
    all_nodes.to_csv(destination+'/'+scenario+'_combined.csv')
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - OUTPUTS COMBINED')

def calc_boykov_kolmogorov(G,source,target,df_nodes_raw,df_edges_raw,scenario,destination):
    R = boykov_kolmogorov(G, source, target,capacity='capacity')

    df_edges_test=located_edges(df_nodes_raw,df_edges_raw,scenario,destination)

    R_edges=nx.to_pandas_edgelist(R)
    R_edges['flow']=R_edges['flow'].abs()
    R_edges=R_edges[R_edges.source<R_edges.target]
    R_edges=R_edges.sort_values(['source', 'target'], ascending=[True, True])
    df_edges_test=df_edges_test.sort_values(['id_node_src', 'id_node_tgt'], ascending=[True, True])

    new_df = pd.merge(df_edges_test, R_edges,  how='left', left_on=['id_node_src','id_node_tgt'], right_on = ['source','target'])

    new_df[(~new_df.source.isna())&(new_df.source<new_df.target) ]

    new_df.to_csv(destination+'/'+scenario+'_edges_bk.csv')
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - BK_EDGES SAVED')
