# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 13:20:25 2016

@author: naus010
"""

# Adjoint test

def adjoint_mcf_test(foh,y):
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
        dMCFi   = dMCFi *  (1 - foh[iyear] * l_mcf_oh - l_mcf_strat - l_mcf_ocean)
        dMCF[iyear] = dMCFi
        dfoh[iyear] = - l_mcf_oh * MCF_save[iyear] * dMCF[iyear]
                        
        # Emissions
        dstock = 0.
        for j in range(i+1 , i+11):
            jyear = j - styear
            if j < edyear: dstock += 0.1 * rapidc[iyear] * dMCF[jyear]
        dfmcf[iyear] = - 0.75 * rapidc[iyear] * dMCF[iyear] + dstock
        if (i + 1) < edyear: dfmcf[iyear] -= -0.25 * rapidc[iyear] * dMCF[iyear+1]
    
    adj_mcf = concatenate((array([dMCFi]), dfmcf))
    return adj_mcf
    
# MCF adjoint test
np.random.seed(0)
x1 = 3+2*np.random.rand(nstate)
Mx = forward_mcf(x1)
x1 = np.concatenate((array([x1[nt]]),x1[nt+1:2*nt+1]))

y = 100 + 100*np.random.rand(nt)

foh_test = x1[:nt]
MTy = adjoint_mcf_test(foh_test,y)

print np.dot(Mx,y)/ np.dot(x1,MTy)

def adjoint_c12_test( foh, y ):
    pulse_12CH4 = y
    em0_12 = em0_c12 / conv_ch4
    
    C12H4_save = ones(nt)
    d12CH4 = zeros(nt)
    df12,dfoh = zeros(nt),zeros(nt)
    d12CH4i = 0
    
    for i in range(edyear-1, styear-1, -1):
        iyear = i-styear
        d12CH4i += pulse_12CH4[iyear]
        
        d12CH4i = d12CH4i * (1 - foh[iyear] * l_ch4_oh - l_ch4_other)
        d12CH4[iyear] = d12CH4i
        dfoh[iyear] = - l_ch4_oh * C12H4_save[iyear] * d12CH4[iyear]
        
        df12[iyear] = em0_12[iyear] * d12CH4[iyear]
        
    adj_c12 = np.concatenate((array([d12CH4i]), df12))
    return adj_c12

# 12CH4 adjoint test
np.random.seed(0)
x1 = 100+2000*np.random.rand(nstate)
Mx = forward_12ch4(x1)
x1 = concatenate((array([x1[2*nt+1]]),x1[2*nt+2:3*nt+2]))

y = 100 + 10*np.random.rand(nt)

foh_test = x1[:nt]
MTy = adjoint_c12_test(foh_test,y)

print dot(Mx,y) / dot(x1,MTy)

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
#        print 'For grid cell',i,'......'
#        if val <= 0.01:
#            print 'Gradient test passed :)'
#        else:
#            print 'Gradient test failed :('
        values.append(val)
    return array( values )*100
    
run_grad_test = True
if run_grad_test:
    # Gradient test 
    print 'Running grad test...'  
    
    # Preparing some data
    xtrue = array([np.random.randint(0,10) for i in range(len(x_prior))])
    x_priorA = xtrue
    #x0_test = np.array([np.random.randint(-10,10) for i in range(2*nx)])
    x0_test = array([xtrue[i]+(1e-1)*(2*np.random.random()-1) for i in range(len(x_prior))])
    # The test
    val = grad_test(x0_test,pert=1e-10)
    
    print 'Grad test mean error (%):' , round(val.mean(),6) , 'and the std of the error:' , round(val.std(),6)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    