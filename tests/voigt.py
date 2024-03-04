import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize as so
import scipy.special
import h5py

class Voigt:
    def __init__(self, energy):
        self.energy = energy

    
    def voigt(self, x, params):
        center, lw, gw = params
        # center : center of Lorentzian line
        # lw : HWFM of Lorentzian (half-width at half-maximum (HWHM))
        # gw : sigma of the gaussian 
        z = (x - center + 1j*lw)/(gw * np.sqrt(2.0))
        w = scipy.special.wofz(z)
        voi = (w.real)/(gw * np.sqrt(2.0*np.pi))
        return voi

    # model
    def MnKalpha(self, x, params, consts=[1,1,1,1,1,1,1,1]):
        # consts = [1,1,1,1,1,1,1,1] : total voigt function
        # consts = [1,0,0,0,0,0,0,0] : each voigt function

        norm, gw, gain = params
        # norm : normalization (counts)
        # gw : sigma of the gaussian 
        # gain : if gain changes (energy shift)

        # constant factor if needed 
        # Mn K alpha lines, Holzer, et al., 1997, Phys. Rev. A, 56, 4554, + an emperical addition
        energy = np.array([ 5898.853, 5897.867, 5894.829, 5896.532, 5899.417, 5902.712, 5887.743, 5886.495])
        lgamma =  np.array([    1.715,    2.043,    4.499,    2.663,    0.969,   1.5528,    2.361,    4.216]) # full width at half maximum
        amp =    np.array([    0.790,    0.264,    0.068,    0.096,   0.0714,   0.0106,    0.372,      0.1])
        prob = (amp * lgamma) / np.sum(amp * lgamma) # probabilites for each lines.

        model_y = 0
        for i, (ene, lg, pr, con) in enumerate(zip(energy, lgamma, prob, consts)):
            voi = self.voigt(x, [ene*gain, lg*0.5, gw])
            model_y += norm * con * pr * voi
        return model_y

    # optimize function
    def calcchi(self, params, model_func, xvalues, yvalues, yerrors):
        model = model_func(xvalues, params)
        chi = (yvalues - model) / yerrors
        return(chi)
    
    # optimize function
    def calclikelihood(self, params, xvalues, yvalues):
        likelihood = 0
        # params[0] = 1
        for i, (xkey, ykey) in enumerate(zip(xvalues, yvalues)):
            p = self.MnKalpha(x=xkey, params=params)
            likelihood += -np.log(p)*ykey
        print("likelihood = " + str(likelihood))
        return likelihood

    
    # optimizer
    def optimizer(self, xvalues, yvalues, yerrors, params_init, model_func, mode):
        if mode == "least_squares":
            param_output = so.least_squares(
                self.calcchi,
                params_init,
                bounds=([0,0,0],[10000,100,100]),
                args=(model_func, xvalues, yvalues, yerrors)
            )
            param_result = param_output.x
            hessian = np.dot(param_output.jac.T, param_output.jac)
            covar_output = np.array(np.linalg.inv(hessian))
            error_result = np.sqrt(covar_output.diagonal())
            dof = len(xvalues) - 1 - len(params_init)
            chi2 = np.sum(np.power(self.calcchi(param_result,model_func,xvalues,yvalues,yerrors),2.0))
            return([param_result, error_result, chi2, dof])

        if mode == "likelihood":
            param_output = so.minimize(
                fun=self.calclikelihood,
                x0=params_init,
                bounds=[(0,10000),(1,50),(0,2)],
                args=(xvalues, yvalues),
                method="l-bfgs-b",
                options={"maxiter":15000}
            )
            param_result = param_output.x
            # covar_output = np.linalg.inv(param_output.hess)
            # error_result = np.sqrt(covar_output.diagonal())
            # dof = len(xvalues) - 1 - len(params_init)
            # chi2 = np.sum(np.power(self.calcchi(param_result,model_func,xvalues,yvalues,yerrors),2.0))
            
            error_result = np.array([0,0,0])
            dof = 0
            chi2 = 0
            return([param_result, error_result, chi2, dof])


    def fit(self, pix, bins=80, params_init=[1800, 10/2.35, 1.001], mode="least_squares"):
        
        # default : maximum likelihood estimation

        # make hist
        rng=[5860, 5940]
        self.y, x = np.histogram(self.energy[pix], bins=bins, range=(rng[0], rng[1]))

        self.x = x[:-1]

        # do fit
        mask = self.y > 10
        self.fit_y = self.y[mask]
        self.fit_x = self.x[mask]
        self.result, self.error, self.chi2, self.dof = self.optimizer(
            xvalues=self.fit_x,
            yvalues=self.fit_y,
            yerrors=np.sqrt(self.fit_y),
            params_init=params_init,
            model_func=self.MnKalpha,
            mode=mode,
        )

        # get results
        self.fwhm = self.result[1]*2.35
        self.fwhm_err = self.error[1]*2.35
        self.energy_shift = self.result[2]
        self.energy_shift_err = self.error[2]
        print("dE = %4.2f +/- %4.2f" % (self.fwhm, self.fwhm_err)+" eV (FWHM)")
        print("Energy shift : " + str(self.result[2]))
        print("Counts : " + str(self.result[0]))


        # plot figure
        fig = plt.figure()
        ax1 = plt.axes([0.1, 0.3, 0.8, 0.6])
        ax2 = plt.axes([0.1, 0.1, 0.8, 0.2], sharex=ax1)

        consts=[1,1,1,1,1,1,1,1]
        ax1.errorbar(self.x, self.y, yerr=np.sqrt(self.y), drawstyle="steps-mid", color="black")
        xx = np.linspace(5860, 5940, 100)
        ax1.plot(xx, self.MnKalpha(xx, self.result, consts=consts), color="red")
        eye = np.eye(len(consts))
        for i, oneeye in enumerate(eye):
            ax1.plot(xx, self.MnKalpha(xx, self.result, consts=oneeye), linestyle="--")
        ax1.grid(linestyle="--", alpha=0.5)
        binwid = (rng[1]-rng[0])/bins
        ax1.set_ylabel("Counts/%.1f eV" % binwid)
        ax1.set_title("Pix.%d, MnK$alpha$(5.9 keV), dE = %4.2f +/- %4.2f" % (pix+1, self.fwhm, self.fwhm_err)+" eV (FWHM)")

        resi = self.y - self.MnKalpha(self.x, self.result, consts=consts)
        ax2.errorbar(self.x, resi, yerr=np.sqrt(self.y), fmt="ko")
        ax2.grid(linestyle="--", alpha=0.5)
        ax2.set_ylabel("Residual")
        ax2.set_xlabel("Energy (eV)")
 
        plt.show()
