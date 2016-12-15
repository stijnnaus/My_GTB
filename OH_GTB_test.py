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
    







































    
    
    
    