import pandas as pd
import numpy as np
from math import degrees,atan,asin,sqrt,pow, atan2
from os import listdir
from os.path import isfile, join
import networkx as nx
import os
import shutil
from datetime import datetime

def dircos2ddd(l, m, n):
    if(m > 0):
        dipdir = (360+degrees(atan(l/m))) % 360
    elif(m < 0):
        dipdir = (540+degrees(atan(l/m))) % 360
    else:
        dipdir = 90
    dip = 90-degrees(asin(n))
    if(dip>90):
        dip=180-dip
        dipdir=dipdir+180
    dipdir=dipdir%360

    return(dip, dipdir)

def pts2dircos(p1x, p1y, p2x, p2y):
    dlsx = p1x-p2x
    dlsy = p1y-p2y
    if(p1x == p2x and p1y == p2y):
        return(0, 0)
    l = dlsx/sqrt((dlsx*dlsx)+(dlsy*dlsy))
    m = dlsy/sqrt((dlsx*dlsx)+(dlsy*dlsy))
    return(l, m)


def ptsdist(p1x, p1y, p2x, p2y):
    dist = sqrt(pow(p1x-p2x, 2)+pow(p1y-p2y, 2))
    return(dist)

def process_surface(surf_path,tri_format,header):
    if(tri_format=='.obj'):
        surface=pd.read_csv(surf_path,sep=' ',names=['type','X','Y','Z'])
        vertices=surface[surface['type']=='v'].drop('type',axis=1)
        triangles=surface[surface['type']=='f'].drop('type',axis=1)
    elif(tri_format=='.ts' or tri_format=='.mx'):
        with open(surf_path) as f:
            datafile = f.readlines()
            for i,line in enumerate(datafile):
                if('TFACE' in line):
                    header=i+1
                    break
            #print('header',header)
        surface=pd.read_csv(surf_path,sep=' ',names=['type','a','b','c','d'],usecols=[0,1,2,3,4],skiprows=header, index_col=False)
        vertices=surface[surface['type']=='VRTX'].drop(['type'],axis=1)
        if(len(vertices)==0):
            vertices=surface[surface['type']=='PVRTX'].drop(['type'],axis=1)
        
        vertices['aint']=vertices.a.astype('int')
        vertices=vertices.set_index('aint')
        vertices=vertices.drop(['a'],axis=1)
        vertices.columns=['X','Y','Z']
        
        atoms=surface[surface['type']=='ATOM'].drop(['type'],axis=1)
        atoms_list={}
        for ind,a in atoms.iterrows():
            #print(vertices.loc[a.b],a.a)
            atoms_list[int(a.a)]={'X':vertices.loc[a.b].X,'Y':vertices.loc[a.b].Y,'Z':vertices.loc[a.b].Z,}
        atoms_list_df=pd.DataFrame.from_dict(atoms_list,orient='index')
        vertices=pd.concat([vertices,atoms_list_df])
    
        triangles=surface[surface['type']=='TRGL'].drop(['type','d'],axis=1)
        
    triangles.columns=['v1','v2','v3']
    triangles['V1']=triangles.v1.astype('int')
    triangles['V2']=triangles.v2.astype('int')
    triangles['V3']=triangles.v3.astype('int')
    triangles=triangles.drop(['v1','v2','v3'],axis=1)
    if(tri_format=='.obj'):
        triangles=triangles-1
        
        
    tri=[triangles.index,vertices.loc[triangles.V1]['X'],vertices.loc[triangles.V1]['Y'],vertices.loc[triangles.V1]['Z'],
     vertices.loc[triangles.V2]['X'],vertices.loc[triangles.V2]['Y'],vertices.loc[triangles.V2]['Z'],
     vertices.loc[triangles.V3]['X'],vertices.loc[triangles.V3]['Y'],vertices.loc[triangles.V3]['Z']]
    
    tri_df=pd.DataFrame(tri)
    tri_df=tri_df.transpose()
    tri_df.columns=['tnum','x1','y1','z1','x2','y2','z2','x3','y3','z3']
    #display(tri_df)
    
    # calc mean orientation of triangles
    p1=np.array([tri_df.x1,tri_df.y1,tri_df.z1])
    p2=np.array([tri_df.x2,tri_df.y2,tri_df.z2])
    p3=np.array([tri_df.x3,tri_df.y3,tri_df.z3])
    p1=p1.T.copy()
    p2=p2.T.copy()
    p3=p3.T.copy()
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    cp=cp.T.copy()
    a, b, c = cp
    d = np.dot(cp, p3) # ax +by +cz=d is eqn of plane

    norm=np.sqrt(a*a+b*b+c*c)
    norm2d=np.sqrt(a*a+b*b)
    
    norm=norm+1e-7 # fix hz planes
    norm2d=norm2d+1e-7 #fix hz planes
    
    l=a/norm
    m=b/norm
    
    l2=a/norm2d
    m2=b/norm2d
    lm2_df=pd.DataFrame(l2,columns=['lsx'])
    lm2_df['lsy']=m2
    lm2_df=lm2_df[~lm2_df.lsx.isna()]

    n=c/norm
    
    l=l[~np.isnan(l).any(), :]
    m=m[~np.isnan(l).any(), :]
    n=n[~np.isnan(l).any(), :]

    lm=l.mean()
    mm=m.mean()
    nm=n.mean()
    
    lm2=lm/sqrt(lm**2+mm**2+nm**2)
    mm2=mm/sqrt(lm**2+mm**2+nm**2)
    nm2=nm/sqrt(lm**2+mm**2+nm**2)
    dip,dipdir=dircos2ddd(lm2,mm2,nm2)
    if(np.isnan(dip)):
        dip=0
    return(dip,dipdir,vertices,lm2_df,tri_df)

def build_strat_surfaces(root_dir,tri_format,upper_padding):
    contacts=pd.DataFrame(columns=['index','X','Y','Z','formation','source'])
    orientations=pd.DataFrame(columns=['X','Y','Z','azimuth','dip','polarity','formation','source'])
    raw_contacts=pd.DataFrame(columns=['X','Y','Z','angle','lsx','lsy','formation','group'])
    fault_contacts=pd.DataFrame(columns=['X','Y','Z','formation'])
    fault_orientations=pd.DataFrame(columns=['X','Y','Z','DipDirection','dip','DipPolarity','formation'])
    fault_dimensions=pd.DataFrame(columns=['Fault','HorizontalRadius','VerticalRadius','InfluenceDistance','incLength','colour'])
    fault_displacements=pd.DataFrame(columns=['X','Y','fname','apparent_displacement','vertical_displacement','downthrow_dir'])
    
    all_sorts=pd.DataFrame(columns=['index','group number','index in group','number in group','code','group','strat_type','supergroup','uctype','colour','meanz'])

    #tri_format='.ts' # 'obj' vs '.ts' or '.mx'

    if(tri_format=='.obj'):
        fault_code='Fault_'
        obj_path_dir=root_dir
        dem_word='dtm'
    else:
        fault_code=['fault_','Fault_']
        obj_path_dir=root_dir
        dem_word='DEM'
        
        

    onlyfiles = [f for f in listdir(obj_path_dir) if isfile(join(obj_path_dir, f))]
    strati=0
    faulti=0
    total_strat=0
    header=0

    xmin=ymin=zmin=1e9
    xmax=ymax=zmax=-1e9

    for file in onlyfiles:
        if (tri_format in file and not any(word in file for word in fault_code) and not dem_word in file):
            total_strat=total_strat+1

    for file in onlyfiles:

        froot=file.replace(tri_format,'')
        if (tri_format in file and any(word in file for word in fault_code)):
            print('process fault',file)
            dip,dipdir,vertices,lm2_df,tri=process_surface(obj_path_dir+file,tri_format,header)
            vertices['formation']=froot
            vmean=[vertices.X.mean(),vertices.Y.mean(),vertices.Z.mean()]
            a_vertex=vertices.sample(1,random_state=1)
            fault_contacts=pd.concat([fault_contacts,vertices])
            fault_orientations.loc[faulti]={'X':a_vertex.iloc[0].X,'Y':a_vertex.iloc[0].Y,'Z':a_vertex.iloc[0].Z,'DipDirection':dipdir,'dip':dip,'DipPolarity':1,'formation':froot}
            
            fault_hr=0.5*sqrt((vertices.X.max()-vertices.X.min())**2+(vertices.Y.max()-vertices.Y.min())**2)#+(vertices.Z.max()-vertices.Z.min())**2)
            fault_vr=fault_hr*2
            fault_id=fault_hr/2
            fault_dimensions.loc[faulti]={'Fault':froot,'HorizontalRadius':fault_hr,'VerticalRadius':fault_vr,'InfluenceDistance':fault_id,'incLength':fault_hr,'colour':'#b07670'}
            fault_displacements.loc[faulti]={'X':vmean[0],'Y':vmean[1],'fname':froot,'apparent_displacement':fault_hr/10000,'vertical_displacement':fault_hr/10000,'downthrow_dir':dipdir}

            faulti=faulti+1
            
            xmin=min(vertices.X.min(),xmin)
            xmax=max(vertices.X.max(),xmax)
            ymin=min(vertices.Y.min(),ymin)
            ymax=max(vertices.Y.max(),ymax)
            zmin=min(vertices.Z.min(),zmin)
            zmax=max(vertices.Z.max(),zmax)        
            
        elif(tri_format in file and not dem_word in file):
            print('process strat',file)
            dip,dipdir,vertices,lm2_df,tri=process_surface(obj_path_dir+file,tri_format,header)
            print(dip,dipdir)
            vertices['index']=0
            vertices['source']='triangulation'
            vertices['formation']=froot
            contacts=pd.concat([contacts,vertices])
            vmean=[vertices.X.mean(),vertices.Y.mean(),vertices.Z.mean()]
            orientations.loc[strati]={'X':vmean[0],'Y':vmean[1],'Z':vmean[2],'azimuth':dipdir,'dip':dip,'polarity':1,'formation':froot,'source':'triangulation'}
            all_sorts.loc[strati]={'index':strati,'group number':0,'index in group':strati,'number in group':total_strat,'code':froot,'group':'all_same','strat_type':'sediment','supergroup':'supergroup_0','uctype':'erode','colour':'#E0CABF','meanz':vertices.Z.mean()}
            lm2_df['angle']=np.degrees(np.arctan2(lm2_df.lsx,lm2_df.lsy))
            lm2_df['X']=(tri.x1+tri.x2+tri.x3)/3
            lm2_df['Y']=(tri.y1+tri.y2+tri.y3)/3
            lm2_df['Z']=(tri.z1+tri.z2+tri.z3)/3
            lm2_df['formation']=froot
            lm2_df['group']='all_same'
            lm2_df=lm2_df[['X','Y','Z','angle','lsx','lsy','formation','group']]
            raw_contacts=pd.concat([raw_contacts,lm2_df])
            strati=strati+1
            
            xmin=min(vertices.X.min(),xmin)
            xmax=max(vertices.X.max(),xmax)
            ymin=min(vertices.Y.min(),ymin)
            ymax=max(vertices.Y.max(),ymax)
            zmin=min(vertices.Z.min(),zmin)
            zmax=max(vertices.Z.max(),zmax)        
            
        elif(tri_format in file  and 'dtm' in file):
            print("don't process dtm",file)

            
        
        
    contacts['index']=range(len(contacts))
    zmax=zmax+upper_padding
    print(xmin,xmax,ymin,ymax,zmin,zmax)
    bbox=create_bbox(xmin,ymin,xmax,ymax,zmin,zmax)
    return(contacts,orientations,raw_contacts,fault_contacts,fault_orientations,fault_dimensions,fault_displacements,all_sorts,bbox,faulti)

def process_dykes_as_faults(obj_path_dir,fault_contacts,fault_orientations,fault_dimensions,fault_displacements,faulti):
    dykes=pd.read_csv(obj_path_dir+'/dykes/DYKES.csv') 
    if('Unnamed: 4' in dykes.columns):
        dykes=dykes.drop(columns=['Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7'])
    dykes['formation']='Fault_dyke_'+dykes.formation.astype('str')
    dykes=dykes[['X','Y','Z','formation']]
    for dyke in dykes.formation.unique():
        adyke=dykes[dykes.formation==dyke]
        bdyke=adyke.copy(deep=True)
        bdyke['Z']=bdyke['Z']-10000.0
        fault_contacts=pd.concat([fault_contacts,adyke,bdyke])
        
        dminx=adyke.X.min()
        dmaxx=adyke.X.max()
        dminy=adyke.Y.min()
        dmaxy=adyke.Y.max()
        length=ptsdist(dminx, dminy, dmaxx, dmaxy)
        l,m=pts2dircos(dminx, dminy, dmaxx, dmaxy)
        dipdir=(180-degrees(atan2(m,-l))) % 360
        
        vmean=[adyke.X.mean(),adyke.Y.mean(),adyke.Z.mean()-5000.0]
        
        fault_orientations.loc[faulti]={'X':vmean[0],'Y':vmean[1],'Z':vmean[2],'DipDirection':dipdir,'dip':90.0,'DipPolarity':1,'formation':dyke}

        fault_hr=0.35*sqrt((adyke.X.max()-adyke.X.min())**2+(adyke.Y.max()-adyke.Y.min())**2)#+(vertices.Z.max()-vertices.Z.min())**2)
        fault_vr=fault_hr*2.5
        fault_id=fault_hr/2
        fault_dimensions.loc[faulti]={'Fault':dyke,'HorizontalRadius':fault_hr*1.5,'VerticalRadius':fault_vr,'InfluenceDistance':fault_id,'incLength':fault_hr,'colour':'#b07670'}
        fault_displacements.loc[faulti]={'X':vmean[0],'Y':vmean[1],'fname':dyke,'apparent_displacement':fault_hr/10000,'vertical_displacement':fault_hr/10000,'downthrow_dir':dipdir}

        faulti=faulti+1
    return(fault_contacts,fault_orientations,fault_dimensions,fault_displacements)

def calculate_formation_thickness(all_sorts):
    all_sorts2=all_sorts.sort_values(by='meanz',ascending=False)
    all_sorts2=all_sorts2.reset_index()
    all_sorts2['index']=all_sorts2.index
    all_sorts2=all_sorts2.drop('level_0',axis=1)
    formation_thickness=pd.DataFrame(columns=['formation','thickness median','thickness std','method'])
    for ind,a_s in all_sorts2[1:].iterrows():
        formation_thickness.loc[ind]={'formation':a_s.code,'thickness median':all_sorts2.loc[ind-1].meanz-a_s.meanz,'thickness std':(all_sorts2.loc[ind-1].meanz-a_s.meanz)/2,'method':'meanz'}
    formation_thickness.loc[0]={'formation':all_sorts2.loc[0].code,'thickness median':10000,'thickness std':10000,'method':'meanz'}

    all_sorts2.loc[0,'supergroup']='supergroup_1'
    all_sorts2.loc[1,'supergroup']='supergroup_1'
    all_sorts2.loc[0,'group']='upper'
    all_sorts2.loc[1,'group']='upper'  
    return(all_sorts2,formation_thickness)  

def calculate_ff_fsg_relationships(fault_dimensions):
    columns=[]
    for ind,f in fault_dimensions.iterrows():
        columns.append(f.Fault)

    ff_arr=np.zeros((len(fault_dimensions),len(fault_dimensions)),dtype=int)
    fault_fault=pd.DataFrame(ff_arr,index=columns,columns=columns)

    gf_arr=np.ones((2,len(fault_dimensions)),dtype=int)
    group_fault=pd.DataFrame(gf_arr,index=['upper','all_same'],columns=columns)

    sgf_arr=np.ones((2,len(fault_dimensions)),dtype=int)
    supergroup_fault=pd.DataFrame(sgf_arr,index=['supergroup_1','supergroup_0'],columns=columns)
    supergroup_fault.index.names = ['supergroup']

    fault_fault_graph= nx.DiGraph()
    for ind,f in fault_dimensions.iterrows():
        fault_fault_graph.add_node(f.Fault)

    f_f_txt=pd.DataFrame(columns=['0'])
    return(f_f_txt)

def create_supergroups():
    super_groups=pd.DataFrame(columns=['0'],index=[0,1])
    super_groups.loc[0]={'0':'upper'}
    super_groups.loc[1]={'0':'all_same'} 
    return(super_groups) 

def create_bbox(xmin,ymin,xmax,ymax,zmin,zmax):
    bbox=pd.DataFrame(columns=['minx','miny','maxx','maxy','lower','upper'])
    bbox.loc[0]={'minx':xmin,'miny':ymin,'maxx':xmax,'maxy':ymax,'lower':zmin,'upper':zmax}
    return(bbox)

def create_paths(root_dir):
    out_dir=root_dir+'/exported/'

    if(not os.path.isdir(out_dir)):
        os.mkdir(out_dir)
    if(not os.path.isdir(out_dir+'/output')):
        os.mkdir(out_dir+'/output')
    if(not os.path.isdir(out_dir+'/dtm')):
        os.mkdir(out_dir+'/dtm')
    if(not os.path.isdir(out_dir+'/tmp')):
        os.mkdir(out_dir+'/tmp')
    if(not os.path.isdir(out_dir+'/graph')):
        os.mkdir(out_dir+'/graph')
    if(not os.path.isdir(out_dir+'/vtk')):
        os.mkdir(out_dir+'/vtk')

def harmonise_data_for_manual_ls(contacts,orientations,formation_thickness,all_sorts2,fault_orientations,fault_contacts,fault_displacements,fault_dimensions,strat_contact_frac):
    contacts_points=contacts.drop(columns=['index','source']).sample(frac=strat_contact_frac,random_state=1)
    contacts_points=contacts_points.rename(columns={"formation": "name"})
    

    stratigraphic_orientations=orientations.copy()
    stratigraphic_orientations=stratigraphic_orientations.rename(columns={"formation": "name"})
    

    stratigraphic_thickness=formation_thickness.drop(columns=['thickness std','method'])
    stratigraphic_thickness=stratigraphic_thickness.rename(columns={"formation": "name","thickness median": "thickness"})
    

    stratigraphic_order=all_sorts2.drop(columns=['index','number in group','strat_type','uctype','colour','meanz'])
    stratigraphic_order=stratigraphic_order.rename(columns={"code": "unit name","index in group": "order"})
    

    fault_orientations=fault_orientations.rename(columns={"formation": "feature_name","DipDirection": "dipdir"})
    

    fault_edges=[]
    

    fault_properties=fault_displacements.drop(columns=['X','Y','apparent_displacement','downthrow_dir'])
    fault_properties=fault_properties.rename(columns={"fname": "Fault","vertical_displacement": "displacement"})
    fault_properties=fault_properties.sort_values(by='Fault')
    fault_properties=fault_properties.set_index('Fault')
    fault_dimensions=fault_dimensions.sort_values(by='Fault')
    fault_dimensions=fault_dimensions.set_index('Fault')
    fault_properties['major_axis']=fault_dimensions.HorizontalRadius*0.90
    fault_properties['intermediate_axis']=fault_dimensions.VerticalRadius
    fault_properties['minor_axis']=fault_dimensions.InfluenceDistance
    

    fault_locations=fault_contacts.rename(columns={"formation": "feature_name"})
    fault_locations['val']=0
    fault_locations['coord']=0
    
    #basements = {'group number': [0,0], 'order': [5,1.5],'unit name':['basement','ubasement'],'group':['all_same','all_same'],'supergroup':['supergroup_0','supergroup_1']}
    #basements_df = pd.DataFrame(data=basements)
    stratigraphic_order['order']=stratigraphic_order.index
    #stratigraphic_order=pd.concat([stratigraphic_order,basements_df])
    stratigraphic_order=stratigraphic_order.sort_values(by='order')
    stratigraphic_order=stratigraphic_order.reset_index()
    stratigraphic_order['order']=stratigraphic_order.index

    return(contacts_points,stratigraphic_orientations,stratigraphic_thickness,stratigraphic_order,fault_orientations,fault_edges,fault_properties,fault_locations)

def import_triangles(root_dir,suffix,strat_contact_frac,upper_padding):
    contacts,orientations,raw_contacts,fault_contacts,fault_orientations,fault_dimensions,fault_displacements,all_sorts,bbox,faulti=build_strat_surfaces(root_dir,suffix,upper_padding)
    fault_contacts,fault_orientations,fault_dimensions,fault_displacements=process_dykes_as_faults(root_dir,fault_contacts,fault_orientations,fault_dimensions,fault_displacements,faulti)
    all_sorts2,formation_thickness=calculate_formation_thickness(all_sorts)
    f_f_txt=calculate_ff_fsg_relationships(fault_dimensions)
    super_groups=create_supergroups()
    create_paths(root_dir)
    contacts_points,stratigraphic_orientations,stratigraphic_thickness,stratigraphic_order,fault_orientations,fault_edges,fault_properties,fault_locations=harmonise_data_for_manual_ls(contacts,orientations,formation_thickness,all_sorts2,fault_orientations,fault_contacts,fault_displacements,fault_dimensions,strat_contact_frac)
    return(contacts_points,stratigraphic_orientations,stratigraphic_thickness,stratigraphic_order,fault_orientations,fault_edges,fault_properties,fault_locations,bbox)


def fake_basement(xmin,xmax,ymin,ymax,zmin,orientations,contacts,all_sorts):

    b_o={'X':xmin+(xmax-xmin)/2, 'Y':ymin+(ymax-ymin)/2, 'Z':zmin-100, 'azimuth':123, 'dip':0.1, 'polarity':1, 'formation':'Basement', 'source':'triangulation'}
    b_o_df=pd.DataFrame(b_o,index=[len(orientations)])


    b_c1={'index':len(contacts), 'X':xmin, 'Y':ymin, 'Z':zmin, 'formation':'Basement', 'source':'triangulation'}
    b_c2={'index':len(contacts)+1, 'X':xmin, 'Y':ymax, 'Z':zmin, 'formation':'Basement', 'source':'triangulation'}
    b_c3={'index':len(contacts)+2, 'X':xmax, 'Y':ymax, 'Z':zmin, 'formation':'Basement', 'source':'triangulation'}
    b_c4={'index':len(contacts)+3, 'X':xmax, 'Y':ymin, 'Z':zmin, 'formation':'Basement', 'source':'triangulation'}
    b_c_df1=pd.DataFrame(b_c1,index=[len(contacts)])
    b_c_df2=pd.DataFrame(b_c2,index=[len(contacts)+1])
    b_c_df3=pd.DataFrame(b_c3,index=[len(contacts)+2])
    b_c_df4=pd.DataFrame(b_c4,index=[len(contacts)+3])
    contacts3=pd.concat([b_c_df1,b_c_df2,b_c_df3,b_c_df4])

    b_as={'index':len(all_sorts), 'group number':'0', 'index in group':len(all_sorts), 'number in group':len(all_sorts), 'code':'Basement',
           'group':'all_same', 'strat_type':'sediment', 'supergroup':'supergroup_0', 'uctype':'erode', 'colour':'#777777', 'meanz':zmin-100}
    b_as_df=pd.DataFrame(b_as,index=[len(all_sorts2)])
    all_sorts2=pd.concat([all_sorts2,b_as_df])

    b_fm_th={'formation':'Basement','thickness median':2000,'thickness std':1000,'method':'meanz'}
    b_fm_th_df=pd.DataFrame(b_fm_th,index=[len(formation_thickness)])
    formation_thickness=pd.concat([formation_thickness,b_fm_th_df])
    return(b_o_df,contacts3,all_sorts2,formation_thickness)

def save_out_like_m2l(out_dir,raw_contacts,all_sorts2,orientations,contacts,basement,fault_displacements,
    fault_orientations,fault_contacts,fault_fault,group_fault,supergroup_fault,fault_dimensions,fault_fault_graph,
    super_groups,bbox,f_f_txt,contacts3,b_o_df,formation_thickness,raw_contact_frac,strat_contact_frac,fault_contact_frac):
    #save out all outputs ready for LoopStructural (with decimation)

    raw_contacts.sample(frac=raw_contact_frac).to_csv(out_dir + '/tmp/raw_contacts.csv',index=False,random_state=1)
    all_sorts2.to_csv(out_dir + '/tmp/all_sorts_clean.csv',index=False)

    orientations.to_csv(out_dir + '/output/orientations_clean.csv',index=False)
    contacts.sample(frac=strat_contact_frac).to_csv(out_dir + '/output/contacts_clean.csv',index=False,random_state=1)

    #add fake basement after decimation
    if(basement):
        orientations2=pd.read_csv(out_dir + '/output/orientations_clean.csv')

        orientations2=pd.concat([orientations2,b_o_df])

        contacts2=pd.read_csv(out_dir + '/output/contacts_clean.csv')
        contacts2=pd.concat([contacts2,contacts3])

        orientations2.to_csv(out_dir + '/output/orientations_clean.csv',index=False)
        contacts2.to_csv(out_dir + '/output/contacts_clean.csv')
        


    fault_displacements.to_csv(out_dir + '/output/fault_displacements3.csv',index=False)
    fault_orientations.to_csv(out_dir + '/output/fault_orientations.csv',index=False)
    fault_contacts.sample(frac=fault_contact_frac).to_csv(out_dir + '/output/faults.csv',index=False,random_state=1)
    fault_fault.to_csv(out_dir + '/output/fault-fault-relationships.csv')
    group_fault.to_csv(out_dir + '/output/group-fault-relationships.csv')
    supergroup_fault.to_csv(out_dir + '/output/supergroup-fault-relationships.csv')
    fault_dimensions.to_csv(out_dir + '/output/fault_dimensions.csv',index=False)
    nx.write_gml(fault_fault_graph,out_dir + '/tmp/fault_network.gml')
    super_groups.to_csv(out_dir + '/tmp/super_groups.csv',header=False,index=False)
    bbox.to_csv(out_dir+'/tmp/bbox.csv',index=False)
    f_f_txt.to_csv(out_dir+'/graph/fault-fault-intersection.txt',header=False,index=False) 
    formation_thickness.to_csv(out_dir + '/output/formation_summary_thicknesses.csv',index=False)