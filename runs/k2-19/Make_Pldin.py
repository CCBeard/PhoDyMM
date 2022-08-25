import sys
import glob

Nplanets = int(sys.argv[1])
mcmcfile = glob.glob('mcmc_*')[0]

def Make_Pldin_from_mcmc(fname, Nplanets):
    '''
    '''
    N = 8 + Nplanets
    
    newpldin = []
    
    with open(fname) as file:

        # loop to read iterate
        # last n lines and print it
        for line in (file.readlines() [-N:]):
            newpldin.append(line)
            
    file.close()
    
    file1 = open('post_demcmc.pldin', "w")

    # Writing to file
    for line in newpldin:
        file1.write(str(line))

    # Closing file
    file1.close()

Make_Pldin_from_mcmc(mcmcfile, Nplanets)
