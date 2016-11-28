# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 13:20:25 2016

@author: naus010
"""

# Adjoint test
    
# MCF adjoint test
if True:
    xbase = 2.*np.random.rand( 4*nt + 3 )
    foh_tes = xbase[:nt]
    mcf_tes = forward_mcf( xbase )
    
    x1 = 5.*np.random.rand( 2*nt + 1 )
    Mx = forward_tl_mcf( x1, foh_tes, mcf_tes )
    
    y = 50-100*np.random.rand(nt)
    MTy0 = adjoint_model_mcf( y, foh_tes, mcf_tes )
    MTy = np.concatenate((MTy0[0],array(MTy0[1]),MTy0[2]))
    
    print 'mcf adjoint test result: ',dot(Mx,y) / dot(x1,MTy)
    #print MTy

# 12CH4 adjoint test
if True:
    xbase = 2*np.random.rand( 4*nt + 3 )
    foh_tes = xbase[:nt]
    c12_tes = forward_c12(xbase)
    
    x1 = 50.*np.random.rand( 4*nt + 3 )
    Mx = forward_tl_c12( x1, foh_tes, c12_tes )
    x1 = np.concatenate((x1[:nt],array([x1[2*nt+1]]),x1[2*nt+2:3*nt+2]))
    
    y = 50-10000*np.random.rand(nt)
    MTy0 = adjoint_model_c12( y, foh_tes, c12_tes )
    MTy = np.concatenate((MTy0[0],array(MTy0[1]),MTy0[2]))
    
    print 'c12 adjoint test result:',dot(Mx,y) / dot(x1,MTy)
    
if True:
    xbase = 2*np.random.rand( 4*nt + 3 )
    foh_tes = xbase[:nt]
    c13_tes = forward_c13(xbase)
    
    x1 = 50.*np.random.rand( 4*nt + 3 )
    Mx = forward_tl_c13( x1, foh_tes, c13_tes )
    x1 = np.concatenate((x1[:nt],array([x1[3*nt+2]]),x1[3*nt+3:4*nt+3]))
    
    y = 50-10000*np.random.rand(nt)
    MTy0 = adjoint_model_c13( y, foh_tes, c13_tes )
    MTy = np.concatenate((MTy0[0],array(MTy0[1]),MTy0[2]))
    
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
            val = abs((predict - reduct)/( ( predict + reduct ) / 2 ))
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
    
    x0_test = array([3.*np.random.random() for i in range(len(x_prior))])
    # The test
    val,deriv,dfohh = grad_test(x0_test,pert=1e-8)
    
    print 'dfoh ratio predict/reduct:', mean(dfohh),std(dfohh)
    
    print 'Grad test mean error (%):' , round(val.mean(),6) , 'and the std of the error:' , round(val.std(),6)
    
    print 'dfoh',val[:nt].mean()
    print 'mcf ini',val[nt]
    print 'dfmcf',val[nt+1:2*nt+1].mean()
    print '12ch4 ini',val[2*nt+1]
    print 'dfc12',val[2*nt+2:3*nt+2].mean()
    print '13ch4 ini',val[3*nt+2]
    print 'dfc13',val[3*nt+3:4*nt+3].mean()
    
    
# Gradient test MCF
    
    
def calculate_J_mcf(x):
    mcf = forward_mcf(x)
    dif = (mcf - mcf_obs)
    dep = dif / mcf_obs_e**2
    cost_list = dif*dep
    J_mcf = sum(cost_list)
    
    J_pri = sum(dot(b_inv2, ( x - x_prior2 )**2)) # prior
    J_tot = .5 * ( J_pri + J_mcf )
    print 'Cost function value:',J_tot
    return J_tot

def calculate_dJdx_mcf(x):
    foh = x[:nt]
    mcf_save = forward_mcf(x)
    
    dif = (mcf_save - mcf_obs)
    dep_mcf = dif / mcf_obs_e**2
    print x_prior2
    
    dfoh,dmcfi,dfmcf = adjoint_model_mcf( dep_mcf, foh, mcf_save )
    
    dJdx_obs = np.concatenate((dfoh,dmcfi,dfmcf))
    dJdx_pri = dot( b_inv2, x - x_prior2 )
    
    dJdx = dJdx_obs + dJdx_pri
    print 'Cost function deriv',max(dJdx)
    return dJdx

def grad_test2(x0,pert = 10**(-5)):
    nx = len(x0)
    x0 = array(x0)
    x_prior2 = x0
    print x_prior2
    deriv = calculate_dJdx_mcf(x0)
    J_prior = calculate_J_mcf(x0)
    
    values = []
    dfohh  = []
    
    for i in range(nx):
        pert_array = zeros(nx)
        pert_array[i] = pert
        x0_pert = x0 + pert_array
        predict = pert*deriv[i]
        J_post = calculate_J_mcf(x0_pert)
        reduct = (J_post - J_prior)
        if i < nt: dfohh.append( predict / reduct )
        if predict == reduct == 0:
            val = 0
        else:
            val = abs((predict - reduct)/( ( predict + reduct ) / 2 ))
#        print 'For grid cell',i,'......'
#        if val <= 0.01:
#            print 'Gradient test passed :)'
#        else:
#            print 'Gradient test failed :('
        values.append(val)
    return array( values )*100,deriv,dfohh

b2a = b[:2*nt+1]
b2  = b2a[:,:2*nt+1]
b_inv2 = linalg.inv(b2)
x_prior2 = x_prior[:2*nt+1]

run_grad_test2 = False
if run_grad_test2:
    # Gradient test 
    print 'Running grad test 2...'  
    
    x0_test2 = array([3.*np.random.random() for i in range(len(x_prior2))])
    x_prior2 = x0_test2
    # The test
    val2,deriv,dfohh = grad_test2(x0_test2,pert=1e-5)
    
    print 'dfoh ratio predict/reduct:', mean(dfohh),std(dfohh)
    
    print 'Grad test mean error (%):' , round(val.mean(),6) , 'and the std of the error:' , round(val.std(),6)
    
    print 'dfoh',val2[:nt].mean()
    print 'mcf ini',val2[nt]
    print 'dfmcf',val2[nt+1:2*nt+1].mean()
    
    

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
    
    
    
    
    
    
    
    
    
    
    
    
    