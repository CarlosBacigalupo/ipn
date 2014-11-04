# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

thisCam = thisStar.exposures.cameras[0]
lambda1, flux1 = clean_flux(thisCam.wavelengths[0], thisCam.red_fluxes[0])
lambda2, flux2 = clean_flux(thisCam.wavelengths[1], thisCam.red_fluxes[1])

# <codecell>

plt.plot(signal.convolve(flux1, flux2[::-1], mode='same'))
plt.plot(signal.fftconvolve(flux1, flux2[::-1], mode='same'))
plt.show()

# <codecell>

plt.plot(np.angle(np.conjugate(np.fft.fft(flux1))*np.fft.fft(flux2)))
plt.show()

# <codecell>

plt.plot(flux1)
plt.plot(flux2)
plt.show()

# <codecell>

ftconv = np.conjugate(np.fft.fft(flux1))*np.fft.fft(flux2)
plt.plot(np.abs(ftconv))
plt.show()

# <codecell>

ww = np.where(np.abs(ftconv) > 15)
plt.plot(ww[0],np.angle(ftconv[ww[0]]),'.')
plt.show()

# <codecell>

test = np.convolve(ftconv,np.ones(4),mode='same')

# <codecell>

ww = np.where(np.abs(test) > 10)
plt.plot(ww[0],np.angle(test[ww[0]]),'.')
plt.show()

# <codecell>

test = np.convolve(ftconv,np.ones(4)/4.0,mode='same')
ww = np.where(np.abs(test) > 10)
plt.plot(ww[0],np.angle(test[ww[0]]),'.')
plt.show()

# <codecell>

x = ww[0][:665]
y = np.angle(test[ww[0]])[:665]

# <codecell>

a = np.polyfit(x, y, 1)
y2 = a[0]*x +a[1]
plt.plot(x,y, '.')
# plt.plot(x, y2)
plt.errorbar(x,y2, y-y2, y-y2)
plt.show()

# <codecell>

plt.plot(np.convolve(np.fft.fft(flux1),np.fft.fft(flux2),'same'))
plt.plot(np.fft.fft(flux1*flux2))
plt.show()

