# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 13:20:25 2016

@author: naus010
"""

# Adjoint test
nxstate = len(x_prior)
nstep = 5
dt = 1./nstep

    


# MCF adjoint test
if True:
    nnrun = 50
    rats = zeros(nnrun)
    for m in range(nnrun):
        xbase = 2.*np.random.rand( nxstate )
        foh_tes = xbase[:nt]
        mcf_tes = forward_mcf( xbase )
        
        x1_0 = 5.*np.random.rand( nxstate )
        Mx = forward_tl_mcf( x1_0, foh_tes, mcf_tes )
        x1 = np.concatenate( ( array([x1_0[0]]), x1_0[3:nt+3], x1_0[nt+3:2*nt+3], x1_0[2*nt+3:3*nt+3] ) )
        
        y = 50-100*np.random.rand(nt)
        MTy0 = adjoint_model_mcf( y, foh_tes, mcf_tes )
        MTy = np.concatenate((array(MTy0[0]),MTy0[1],MTy0[2],MTy0[3]))
    
        rats[m] = dot(Mx,y) / dot(x1,MTy)
    
    print 'mcf adjoint test result: ',dot(Mx,y) / dot(x1,MTy)
    #print MTy

# 12CH4 adjoint test
if True:
    xbase = 2*np.random.rand( nxstate )
    foh_tes = xbase[:nt]
    c12_tes = forward_c12(xbase)
    
    x1 = 50.*np.random.rand( nxstate )
    Mx = forward_tl_c12( x1, foh_tes, c12_tes )
    x1 = np.concatenate(( array([x1[1]]), x1[3:nt+3], x1[3*nt+3:4*nt+3] ))
    
    y = 50-10000*np.random.rand(nt)
    MTy0 = adjoint_model_c12( y, foh_tes, c12_tes )
    MTy = np.concatenate(( array(MTy0[0]), MTy0[1],MTy0[2]))
    
    print 'c12 adjoint test result:',dot(Mx,y) / dot(x1,MTy)

# 13CH4 adjoint test
if True:
    xbase = 2*np.random.rand( nxstate )
    foh_tes = xbase[:nt]
    c13_tes = forward_c13(xbase)
    
    x1 = 50.*np.random.rand( nxstate )
    Mx = forward_tl_c13( x1, foh_tes, c13_tes )
    x1 = np.concatenate(( array([x1[2]]), x1[3:nt+3], x1[4*nt+3:5*nt+3] ))
    
    y = 50-10000*np.random.rand(nt)
    MTy0 = adjoint_model_c13( y, foh_tes, c13_tes )
    MTy = np.concatenate(( array(MTy0[0]), MTy0[1], MTy0[2] ))
    
    print 'test result c13 adjoint:',dot(Mx,y) / dot(x1,MTy)
    


# Gradient test

def grad_test(x0,pert = 10**(-5)):
    nx = len(x0)
    x0 = array(x0)
    x0p = state_to_precon(x0)
    deriv = calculate_dJdx(x0p)
    J_prior = calculate_J(x0p)
    
    values = []
    dfohh  = []
    
    for i in range(nx):
        pert_array = zeros(nx)
        pert_array[i] = pert
        x0p_pert = x0p + pert_array
        predict = pert*deriv[i]
        J_post = calculate_J(x0p_pert)
        reduct = (J_post - J_prior)
        if i < nt: dfohh.append( predict / reduct )
        if predict == reduct == 0:
            val = 0
        else:
            val = predict/ reduct 
#        print 'For grid cell',i,'......'
#        if val <= 0.01:
#            print 'Gradient test passed :)'
#        else:
#            print 'Gradient test failed :('
        values.append(val)
    return array( values )*100,deriv,dfohh
    
run_grad_test = True
if run_grad_test:
    # Gradient test 
    print 'Running grad test...'  
    
    #x0_test = array([3.*np.random.random() for i in range(len(x_prior))])
    x0_test = x_opt + .1*np.random.rand(nstate)
    # The test
    val,deriv,dfohh = grad_test(x0_test,pert=1e-7)
    
    print 'dfoh ratio predict/reduct:', mean(dfohh),std(dfohh)
    
    print 'Grad test mean error (%):' , round(val.mean(),6) , 'and the std of the error:' , round(val.std(),6)
    
    print 'dfoh',val[3:nt+3].mean()
    print 'mcf ini',val[0]
    print 'dfslow',val[2*nt+3:3*nt+3].mean()
    print 'dfstock',val[nt+3:2*nt+3].mean()
    print '12ch4 ini',val[1]
    print 'dfc12',val[3*nt+2:4*nt+2].mean()
    print '13ch4 ini',val[2]
    print 'dfc13',val[4*nt+3:5*nt+3].mean()
    
# Gradient test MCF

def calculate_J_mcf(x):
    mcf = forward_mcf(x)
    dif = (mcf - mcf_obs)
    dep = dif / mcf_obs_e**2
    J_obs = sum( dep * dif )
    
    J_pri = sum( dot(b_inv, ( x - x_prior )**2) ) # prior
    J_tot = .5 * ( J_pri + J_obs )
    print 'Cost function value:',J_tot
    
    return J_tot, .5*J_obs, .5*J_pri

def calculate_dJdx_mcf(x):
    _, _, _, foh, _, _, _, _ = unpack(x)
    mcf_save = forward_mcf(x)
    
    dif = (mcf_save - mcf_obs)
    dep_mcf = dif / mcf_obs_e**2
    
    dmcf0,dfoh_mcf,dfst,dfsl = adjoint_model_mcf( dep_mcf, foh, mcf_save )
    dfoh = dfoh_mcf
    
    dJdx_obs = np.concatenate(( dmcf0, dfoh, \
                                dfst, dfsl ))
    dJdx_pri = dot(b_inv, (x-x_prior))
    dJdx_pri_mcf = concatenate(( array([dJdx_pri[0]]), dJdx_pri[3:nt+3], \
                                dJdx_pri[nt+3:2*nt+3], dJdx_pri[2*nt+3:3*nt+3] ))
    
    dJdx = dJdx_obs + dJdx_pri_mcf
    print 'Cost function deriv:',max(dJdx)
    return dJdx, dJdx_obs, dJdx_pri_mcf
    
def grad_test_mcf(x0,pert = 10**(-5)):
    nx = len(x0)
    x0 = array(x0)
    deriv,deriv_obs,deriv_bg = calculate_dJdx_mcf(x0)
    J_prior,J_probs,J_prbg = calculate_J_mcf(x0)
    
    values = []
    vals_obs,vals_bg = [],[]
    dfohh  = []
    
    mcf_loc = [0] + range(3,3*nt+3)
    
    
    for i in range(3*nt+1):
        pert_array = zeros(nx)
        pert_array[ mcf_loc[i] ] = pert
        x0_pert = x0 + pert_array
        predict = pert*deriv[i]
        pred_obs,pred_bg = pert*deriv_obs[i],pert*deriv_bg[i]
        J_post,J_ptobs,J_ptbg = calculate_J_mcf(x0_pert)
        reduct = J_post - J_prior
        redu_obs,redu_bg = J_ptobs - J_probs, J_ptbg - J_prbg
        #print 'ratio obs:', pred_obs,redu_obs
        #print 'ratio bg: ', pred_bg,redu_bg
        print 'J bg dif',J_prbg - J_ptbg
        print 'Jobs dif',J_probs - J_ptobs
        #print 'dJdx    ',deriv_bg[i]
        if 0 < i < nt+1: dfohh.append( predict / reduct )
        if predict == reduct == 0:
            val = 0
        else:
            val = abs((predict - reduct)/( ( predict + reduct ) / 2 ))
        
        if pred_obs == redu_obs == 0:
            val_obs = 0
        else:
            val_obs = abs((pred_obs - redu_obs)/( ( pred_obs + redu_obs ) / 2 ))
        
        if pred_bg == redu_bg == 0:
            val_bg = 0
        else:
            val_bg = abs((pred_bg - redu_bg)/( ( pred_bg + redu_bg ) / 2 ))
            
        values.append(val)
        vals_obs.append(val_obs); vals_bg.append(val_bg)
    return array( values )*100, deriv, dfohh, array( vals_obs )*100, array( vals_bg )*100
    
run_grad_test_mcf = False
if run_grad_test_mcf:
    # Gradient test 
    print 'Running grad test MCF...'  
    
    x0_test = array([3.*np.random.random() for i in range(len(x_prior))])
    #x0_test = deepcopy(x_prior)
    # The test
    val,deriv,dfohh,val_obs,val_bg = grad_test_mcf(x0_test,pert=1e-8)
    
    print 'dfoh ratio predict/reduct:', mean(dfohh),std(dfohh)
    
    print 'Grad test mean error (%):' , round(val.mean(),6) , 'and the std of the error:' , round(val.std(),6)
    
    print 'OBSERVATIONS'
    print 'dfoh',val_obs[1:nt+1].mean()
    print 'mcf ini',val_obs[0]
    print 'dfslow',val_obs[2*nt+1:3*nt+1].mean()
    print 'dfstock',val_obs[nt+1:2*nt+1].mean()
    
    print 'BACKGROUND'
    print 'dfoh',val_bg[1:nt+1].mean()
    print 'mcf ini',val_bg[0]
    print 'dfslow',val_bg[2*nt+1:3*nt+1].mean()
    print 'dfstock',val_bg[nt+1:2*nt+1].mean()
    
    print 'BOTH'
    print 'dfoh',val[1:nt+1].mean()
    print 'mcf ini',val[0]
    print 'dfslow',val[2*nt+1:3*nt+1].mean()
    print 'dfstock',val[nt+1:2*nt+1].mean()
    
    
    
# Gradient test 12CH4

def calculate_J_c12(x):
    c12 = forward_c12(x)
    dif = (c12 - c12_obs)
    dep = dif / c12_obs_e**2
    J_obs = sum( dep * dif )
    
    J_pri = sum( dot(b_inv, ( x - x_prior )**2) ) # prior
    J_tot = .5 * ( J_pri + J_obs )
    print 'Cost function value:',J_tot
    
    return J_tot, .5*J_obs, .5*J_pri

def calculate_dJdx_c12(x):
    _, _, _, foh, _, _, _, _ = unpack(x)
    c12_save = forward_c12(x)
    
    dif = (c12_save - c12_obs)
    dep_c12 = dif / c12_obs_e**2
    
    dc120,dfoh_c12,df12 = adjoint_model_c12( dep_c12, foh, c12_save )
    dfoh = dfoh_c12
    
    dJdx_obs = np.concatenate(( dc120, dfoh, df12 ))
    dJdx_pri = dot( b_inv, (x-x_prior) )
    dJdx_pri_c12 = concatenate(( array([dJdx_pri[1]]), dJdx_pri[3:nt+3], dJdx_pri[3*nt+3:4*nt+3] ))
    
    dJdx = dJdx_obs + dJdx_pri_c12
    print 'Cost function deriv:',max(dJdx)
    return dJdx, dJdx_obs, dJdx_pri_c12
    
def grad_test_c12(x0,pert = 10**(-5)):
    nx = len(x0)
    x0 = array(x0)
    deriv,deriv_obs,deriv_bg = calculate_dJdx_c12(x0)
    J_prior,J_probs,J_prbg = calculate_J_c12(x0)
    
    values = []
    vals_obs,vals_bg = [],[]
    dfohh  = []
    
    c12_loc = [1] + range(3, nt+3) + range(3*nt+3, 4*nt+3)
    
    
    for i in range(2*nt+1):
        pert_array = zeros(nx)
        pert_array[ c12_loc[i] ] = pert
        x0_pert = x0 + pert_array
        predict = pert*deriv[i]
        pred_obs,pred_bg = pert*deriv_obs[i],pert*deriv_bg[i]
        J_post,J_ptobs,J_ptbg = calculate_J_c12(x0_pert)
        reduct = J_post - J_prior
        redu_obs,redu_bg = J_ptobs - J_probs, J_ptbg - J_prbg
        #print 'ratio obs:', pred_obs,redu_obs
        #print 'ratio bg: ', pred_bg,redu_bg
        #print 'J bg dif',J_prbg - J_ptbg
        #print 'Jobs dif',J_probs - J_ptobs
        #print 'dJdx    ',deriv_bg[i]
        if 0 < i < nt+1: dfohh.append( predict / reduct )
        if predict == reduct == 0:
            val = 0
        else:
            val = abs((predict - reduct)/( ( predict + reduct ) / 2 ))
        
        if pred_obs == redu_obs == 0:
            val_obs = 0
        else:
            val_obs = abs((pred_obs - redu_obs)/( ( pred_obs + redu_obs ) / 2 ))
        
        if pred_bg == redu_bg == 0:
            val_bg = 0
        else:
            print pred_bg,redu_bg
            val_bg = abs((pred_bg - redu_bg)/( ( pred_bg + redu_bg ) / 2 ))
            
        values.append(val)
        vals_obs.append(val_obs); vals_bg.append(val_bg)
    return array( values )*100, deriv, dfohh, array( vals_obs )*100, array( vals_bg )*100
    
run_grad_test_c12 = False
if run_grad_test_c12:
    # Gradient test 
    print 'Running grad test C12...'  
    
    x0_test = array([3.*np.random.random() for i in range(len(x_prior))])
    #x0_test = deepcopy(x_prior)
    # The test
    val,deriv,dfohh,val_obs,val_bg = grad_test_c12(x0_test,pert=1e-15)
    
    print 'dfoh ratio predict/reduct:', mean(dfohh),std(dfohh)
    
    print 'Grad test mean error (%):' , round(val.mean(),6) , 'and the std of the error:' , round(val.std(),6)
    
    print 'OBSERVATIONS'
    print 'dfoh',val_obs[1:nt+1].mean()
    print 'c12 ini',val_obs[0]
    print 'df12',val_obs[nt+1:2*nt+1].mean()
    
    print 'BACKGROUND'
    print 'dfoh',val_bg[1:nt+1].mean()
    print 'c12 ini',val_bg[0]
    print 'df12',val_bg[nt+1:2*nt+1].mean()
    
    print 'BOTH'
    print 'dfoh',val[1:nt+1].mean()
    print 'c12 ini',val[0]
    print 'df12',val[nt+1:2*nt+1].mean()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

#    
#def forward_tl_c12_test(x, foh, c12_save):
#    dfoh, dc12_0, dfc12 = x[:nt], x[2*nt+1], x[2*nt+2 : 3*nt+2]
#    dem0_12 = dfc13 * (em0_12 / conv_c12)
#    
#    dc12s = []; dc12 = dc12_0
#    for year in range(styear,edyear):
#        i = year - styear
#        dc12 += dfc12[i] * dem0_12[i]
#        dc12  = dc12 * ( 1 - l_ch4_other - l_ch4_oh * foh[i]) - \
#                dfoh[i] * c12_save * l_ch4_oh
#        dc12s.append(dc12)
#    
#    return array(dc12s)
#
#def adjoint_model_c12( y, foh, C12H4_save ):
#    pulse_12CH4 = y
#    em0_12 = em0_c12 / conv_ch4
#    
#    df12,dfoh = zeros(nt),zeros(nt)
#    d12CH4i = 0
#    
#    for i in range(edyear-1, styear-1, -1):
#        iyear = i-styear
#        d12CH4i += pulse_12CH4[iyear]
#        
#        dfoh[iyear] = - l_ch4_oh * C12H4_save[iyear] * d12CH4i
#        d12CH4i = d12CH4i * (1 - foh[iyear] * l_ch4_oh - l_ch4_other)
#        
#        df12[iyear] = em0_12[iyear] * d12CH4i
#        
#    adj_c12 = concatenate((dfoh,array([d12CH4i]), df12))
#    return adj_c12
    

    
    
'''
    
# igac mcf forward vs mine
    
mcf_me = forward_mcf(x_prior)
_,_,_,mcf_he_d = run_model_mcf(state_apri)
mcf_he_d = array( [ array(mcf_he_d[i]) / conv_mcf for i in range(len(mcf_he_d)) ] )
mcf_he_m = array( [ mean(array(mcf_he_d[i])) for i in range(len(mcf_he)) ] )
mcf_he_ms = split( mcf_he_m , len(mcf_he_m)/12 )
mcf_he_y = array( [ mean( array(mcf_he_ms[i]) ) for i in range(len(mcf_he_ms)) ] )


xyr = linspace(1988,2008,num=len(mcf_he))

plt.plot(range(1988,2009),mcf_me, 'ro-')
plt.plot(range(1988,2009),mcf_he_y, 'go-')

'''
    
    
    
    