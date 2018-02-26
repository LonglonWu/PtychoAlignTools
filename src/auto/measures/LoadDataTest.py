import numpy as np
import tifffile


def LoadData2(rot90=True):
    x_pos = np.asarray([   0.,    0.,    0.,    0.,  281.,  281.,  281.,  281.,  533., 533.,  533.,  533.])
    y_pos = np.asarray([ -15., -242., -469., -696.,    0., -227., -454., -681.,    0., -227., -454., -681.])

    maskPath = '/home/francesco/Seafile/FGpty/fra_2-3/ilum2.tif'
    mask = tifffile.imread(maskPath)
    if rot90:
        mask = np.rot90(mask) #mask

    NIMG = 12
    basePath = '/home/francesco/Seafile/FGpty/CoEnAfter_0/fra_ca_single_En15_parallelFalse_1516202311/m/'
    filenamePart1 = 'mag_784.0eV_0268.00mu_fd0.3050000-y'
    filenamePart2 = '.000000-x0.000000.tiff'

    testTable = []
    testTable.append(mask)

    testidx = []

    testidx.append([0,1])
    testidx.append([1,2])
    testidx.append([2,3])

    testidx.append([4,5])
    testidx.append([5,6])
    testidx.append([6,7])

    testidx.append([8,9])
    testidx.append([9,10])
    testidx.append([10,11])


    for i in range(len(testidx)): #populate table
        testi = []
        fni = []
        for k in range(2):
            fni.append(basePath + filenamePart1 + str(testidx[i][k]) + filenamePart2)
        idxnow1 = testidx[i][0]
        idxnow2 = testidx[i][1]
        xpostmp = [x_pos[idxnow1], x_pos[idxnow2]]
        ypostmp = [y_pos[idxnow1], y_pos[idxnow2]]
        testi.append(fni)
        testi.append(xpostmp)
        testi.append(ypostmp)
        testTable.append(testi.copy())
    return testTable

def LoadData():

	testTable = []
	maskPath = '/home/francesco/Seafile/FGpty/fra_2-3/ilum2.tif'
	mask = ( tifffile.imread( '/home/francesco/Seafile/FGpty/fra_2-3/ilum2.tif' ) ) #mask
	testTable.append(mask)


	test1 = []
	fn1 = [
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y0.000000-x0.000000.tiff',
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y1.000000-x0.000000.tiff'
		]
	xpos1 = np.asarray([0, -235])
	ypos1 = np.asarray([0, 0])
	test1.append(fn1)
	test1.append(xpos1)
	test1.append(ypos1)
	testTable.append(test1)

	test2 = []
	fn2 = [
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y1.000000-x0.000000.tiff',
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y2.000000-x0.000000.tiff'
			]
	xpos2 = np.asarray([-235, -460])
	ypos2 = np.asarray([0, 0])
	test2.append(fn2)
	test2.append(xpos2)
	test2.append(ypos2)
	testTable.append(test2)

	test3 = []
	fn3 = [
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y2.000000-x0.000000.tiff',
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y3.000000-x0.000000.tiff'
			]
	xpos3 = np.asarray([-460, -600])
	ypos3 = np.asarray([0, -5])
	test3.append(fn3)
	test3.append(xpos3)
	test3.append(ypos3)
	testTable.append(test3)


################

	test4 = []
	fn4 = [
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y4.000000-x0.000000.tiff',
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y5.000000-x0.000000.tiff'
			]
	xpos4 = np.asarray([15, -210])
	ypos4 = np.asarray([-280, -280])
	test4.append(fn4)
	test4.append(xpos4)
	test4.append(ypos4)
	testTable.append(test4)

	test5 = []
	fn5 = [
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y5.000000-x0.000000.tiff',
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y6.000000-x0.000000.tiff'
		]
	xpos5 = np.asarray([-210, -446])
	ypos5 = np.asarray([-280, -285])
	test5.append(fn5)
	test5.append(xpos5)
	test5.append(ypos5)
	testTable.append(test5)

	test6 = []
	fn6 = [
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y6.000000-x0.000000.tiff',
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y7.000000-x0.000000.tiff'
		]
	xpos6 = np.asarray([-446, -670])
	ypos6 = np.asarray([-258, -281])
	test6.append(fn6)
	test6.append(xpos6)
	test6.append(ypos6)
	testTable.append(test6)

#################


	test7 = []
	fn7 = [
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y8.000000-x0.000000.tiff',
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y9.000000-x0.000000.tiff'
		]
	xpos7 = np.asarray([15, -210])
	ypos7 = np.asarray([-539, -540])
	test7.append(fn7)
	test7.append(xpos7)
	test7.append(ypos7)
	testTable.append(test7)

	test8 = []
	fn8 = [
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y9.000000-x0.000000.tiff',
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y10.000000-x0.000000.tiff'
		]
	xpos8 = np.asarray([-210, -434])
	ypos8 = np.asarray([-540, -541])
	test8.append(fn8)
	test8.append(xpos8)
	test8.append(ypos8)
	testTable.append(test8)

	test9 = []
	fn9 = [
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y10.000000-x0.000000.tiff',
			'/home/francesco/Seafile/PHD/imageAlign/imgs_orig32bit_orig8bit_outFeat/imagesProbeAlign_francescoEn03_parallelFalse_1493985916/m/mag_778.0eV_0288.00mu_fd0.3050000-y11.000000-x0.000000.tiff'
		]
	xpos9 = np.asarray([-434, -669])
	ypos9 = np.asarray([-541, -540])
	test9.append(fn9)
	test9.append(xpos9)
	test9.append(ypos9)
	testTable.append(test9)

	return testTable
