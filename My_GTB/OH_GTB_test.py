# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 13:20:25 2016

@author: naus010
"""

# Adjoint test

def adjoint_mcf_test(foh, MCF_save, y):
    pulse_MCF = y
    MCF_save = Mx*5000
    dMCF= zeros(nt)
    dfmcf,dfoh = zeros(nt),zeros(nt)
    dMCFi = 0
    
    rapidc = rapid/conv_mcf
    
    for i in range(edyear-1,styear-1,-1):
        iyear = i-styear
        
        # Add adjoint pulses
        dMCFi += pulse_MCF[iyear]
        
        # Chemistry
        dfoh[iyear] = - l_mcf_oh * MCF_save[iyear] * dMCFi
        dMCFi   = dMCFi *  (1 - foh[iyear] * l_mcf_oh - l_mcf_strat - l_mcf_ocean)
        dMCF[iyear] = dMCFi
                        
        # Emissions
        dstock = 0.
        for j in range(i+1 , i+11):
            jyear = j - styear
            if j < edyear: dstock += 0.1 * rapidc[iyear] * dMCF[jyear]
        dfmcf[iyear] = - 0.75 * rapidc[iyear] * dMCF[iyear] + dstock
        if (i + 1) < edyear: dfmcf[iyear] -= -0.25 * rapidc[iyear] * dMCF[iyear+1]
    
    adj_mcf = concatenate(dfoh, (array([dMCFi]), dfmcf))
    return adj_mcf
    
# MCF adjoint test
if False:
    np.random.seed(0)
    x1 = 3+2*np.random.rand( 2*nt + 1 )
    Mx = forward_mcf(x1)
    x1 = np.concatenate((array([x1[nt]]),x1[nt+1:2*nt+1]))
    
    y = 100 + 100*np.random.rand(nt)
    
    foh_test = x1[:nt]
    MTy = adjoint_mcf_test(foh_test,ones(nt),y)
    
    print np.dot(Mx,y)/ np.dot(x1,MTy)

def adjoint_model_c12_tes( dep, foh, c12_save ):
    pulse_c12 = dep
    em0_12 = em0_c12 / conv_ch4
    
    df12,dfoh = zeros(nt),zeros(nt)
    dc12i = 0
    
    for i in range(edyear-1, styear-1, -1):
        iyear = i-styear
        dc12i += pulse_c12[iyear]
        
        dfoh[iyear] = - l_ch4_oh * c12_save[iyear] * dc12i
        dc12i = dc12i * (1 - foh[iyear] * l_ch4_oh - l_ch4_other)
        
        df12[iyear] = em0_12[iyear] * dc12i
        
    adj_c12 = np.concatenate((dfoh, array([dc12i]), df12))
    return adj_c12

# 12CH4 adjoint test
if True:
    np.random.seed(0)
    xbase = 2*np.random.rand( 4*nt + 3 )
    foh_tes = xbase[:nt]
    c12_tes = forward_c12(xbase)
    
    x1 = 5*np.random.rand( 4*nt + 3 )
    Mx = forward_tl_c12( x1, foh_tes, c12_tes )
    x1 = np.concatenate((x1[:nt],array([x1[2*nt+1]]),x1[2*nt+2:3*nt+2]))
    
    y = 100 + 10*np.random.rand(nt)
    MTy = adjoint_model_c12_tes( y, foh_tes, c12_tes )
    
    print dot(Mx,y) , dot(x1,MTy)

# Gradient test

def grad_test(x0,pert = 10**(-5)):
    nx = len(x0)
    x0 = array(x0)
    x_prior = x0
    deriv = calculate_dJdx(x0)
    J_prior = calculate_J(x0)
    
    values = []
    
    for i in range(nx):
        pert_array = zeros(nx)
        pert_array[i] = pert
        x0_pert = x0 + pert_array
        
        predict = pert*deriv[i]
        J_post = calculate_J(x0_pert)
        reduct = (J_post - J_prior)
        if predict == reduct == 0:
            val = 0
        else:
            val = abs((predict - reduct)/( ( predict + reduct ) / 2 ))
        print 'from deriv: ', predict
        print 'from perturb: ',reduct
#        print 'For grid cell',i,'......'
#        if val <= 0.01:
#            print 'Gradient test passed :)'
#        else:
#            print 'Gradient test failed :('
        values.append(val)
    return array( values )*100,deriv
    
run_grad_test = False
if run_grad_test:
    # Gradient test 
    print 'Running grad test...'  
    
    # Preparing some data
    xtrue = array([np.random.randint(0,10) for i in range(len(x_prior))])
    x_priorA = xtrue
    #x0_test = np.array([np.random.randint(-10,10) for i in range(2*nx)])
    x0_test = array([xtrue[i]+(1e-1)*(2*np.random.random()-1) for i in range(len(x_prior))])
    # The test
    val,deriv = grad_test(x0_test,pert=1e-5)
    
    print 'Grad test mean error (%):' , round(val.mean(),6) , 'and the std of the error:' , round(val.std(),6)
    
    print 'dfoh',val[:nt].mean()
    print 'mcf ini',val[nt]
    print 'dfmcf',val[nt+1:2*nt+1].mean()
    print '12ch4 ini',val[2*nt+1]
    print 'dfc12',val[2*nt+2:3*nt+2].mean()
    print '13ch4 ini',val[3*nt+2]
    print 'dfc13',val[3*nt+3:4*nt+3].mean()
    
    
    
    
    
    

    
    
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    