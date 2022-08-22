import numpy as np
import pandas as pd


def Create_New_Pldin(posteriors, newpldin, npl):


    letters = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k',]
    
    post = pd.read_csv(posteriors, sep='\t', header=None)
    #write the first line of the new file
    outlines = ['planet         period (d)               T0 (d)                  e                   i (deg)                 Omega (deg)               omega(deg)               mp (mjup)              rpors']
    for n in range(npl):
        letter = letters[n]
        
        secos = float(post[1][np.where(post[0] == '$\\sqrt{e}\\cos \\omega$$_'+str(letter)+'$')[0]].to_numpy(dtype=str)[0].split('+')[0])
        sesin = float(post[1][np.where(post[0] == '$\\sqrt{e}\\sin \\omega$$_'+str(letter)+'$')[0]].to_numpy(dtype=str)[0].split('+')[0])
        omeg = np.arctan(sesin/secos)
        #convert to degrees
        
        line = ['0.{}'.format(n+1),
               '\t',
               post[1][np.where(post[0] == 'Period$_{}$      '.format(letter))[0]].to_numpy(dtype=str)[0].split('+')[0],
               post[1][np.where(post[0] == 'T$_0,$$_{}$      '.format(letter))[0]].to_numpy(dtype=str)[0].split('+')[0],
               post[1][np.where(post[0] == '$e_{}$           '.format(letter))[0]].to_numpy(dtype=str)[0].split('+')[0],
               post[1][np.where(post[0] == 'i$_{}$           '.format(letter))[0]].to_numpy(dtype=str)[0].split('+')[0],
               post[1][np.where(post[0] == '$\\Omega$$_{}$    '.format(letter))[0]].to_numpy(dtype=str)[0].split('+')[0],
                str(np.round(omeg,15)),
               post[1][np.where(post[0] == 'M$_{jup,}$$_'+str(letter)+'$  '.format(letter))[0]].to_numpy(dtype=str)[0].split('+')[0],
               post[1][np.where(post[0] == 'R$_p$/R$_s$$_{}$ '.format(letter))[0]].to_numpy(dtype=str)[0].split('+')[0],
               ]
        
        outlines.append("".join(line))
        
    #finish up with stellar parameters
    outlines.append(float(post[1][np.where(post[0] == 'M$_s$           ')[0]].to_numpy(dtype=str)[0].split('+')[0]))
    outlines.append(float(post[1][np.where(post[0] == 'R$_s$           ')[0]].to_numpy(dtype=str)[0].split('+')[0]))
    outlines.append(float(post[1][np.where(post[0] == 'c$_1$           ')[0]].to_numpy(dtype=str)[0].split('+')[0]))
    outlines.append(float(post[1][np.where(post[0] == 'c$_2$           ')[0]].to_numpy(dtype=str)[0].split('+')[0]))
    outlines.append(float(post[1][np.where(post[0] == 'dilute          ')[0]].to_numpy(dtype=str)[0].split('+')[0]))

    file1 = open(newpldin, "w")

    # Writing to file
    for line in outlines:
        file1.write(str(line)+'\n')

    # Closing file
    file1.close()
        
    
    
    
Create_New_Pldin('analysis_dir/fits_1sigma_kepler36_longcadence.txt', 'post_demcmc.pldin', 2)
