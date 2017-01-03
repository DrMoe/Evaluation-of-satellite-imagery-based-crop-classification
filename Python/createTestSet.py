from osgeo import gdal,ogr
import csv
import struct

pathShape = ''

# Raster File
src_filename = ''

for i in range(1,15):

    samples = i*30

    filename = str(samples) + ''

    # Shape File
    shp_filename = pathShape + filename + '.shp'

    csv_filename = filename + '' + '.csv'

    index_1 = []
    index_2 = []
    index_3 = []
    index_4 = []
    index_5 = []
    index_6 = []
    index_7 = []
    index_8 = []
    index_9 = []
    index_10 = []
    index_11 = []
    index_12 = []
    index_13 = []
    index_14 = []
    index_15 = []
    index_16 = []
    index_17 = []
    index_18 = []
    index_19 = []
    index_20 = []

    kode = []
    markkode = []

    src_ds=gdal.Open(src_filename)
    gt=src_ds.GetGeoTransform()
    rb_1=src_ds.GetRasterBand(1)

    rb_2 = src_ds.GetRasterBand(2)
    rb_3 = src_ds.GetRasterBand(3)
    rb_4 = src_ds.GetRasterBand(4)
    rb_5 = src_ds.GetRasterBand(5)
    rb_6 = src_ds.GetRasterBand(6)
    rb_7 = src_ds.GetRasterBand(7)
    rb_8 = src_ds.GetRasterBand(8)
    rb_9 = src_ds.GetRasterBand(9)
    rb_10 = src_ds.GetRasterBand(10)
    rb_11 = src_ds.GetRasterBand(11)
    rb_12 = src_ds.GetRasterBand(12)
    rb_13 = src_ds.GetRasterBand(13)
    rb_14 = src_ds.GetRasterBand(14)
    rb_15 = src_ds.GetRasterBand(15)
    rb_16 = src_ds.GetRasterBand(16)
    rb_17 = src_ds.GetRasterBand(17)
    rb_18 = src_ds.GetRasterBand(18)
    rb_19 = src_ds.GetRasterBand(19)
    rb_20 = src_ds.GetRasterBand(20)

    ds=ogr.Open(shp_filename)
    lyr=ds.GetLayer()
    for feat in lyr:
        geom = feat.GetGeometryRef()
        afgkode = feat.GetField("AfgKode")
        mkkode = feat.GetField("MarkblokNr")

        mx,my=geom.GetX(), geom.GetY()  #coord in map units

        #Convert from map to pixel coordinates.
        #Only works for geotransforms with no rotation.
        px = int((mx - gt[0]) / gt[1]) #x pixel
        py = int((my - gt[3]) / gt[5]) #y pixel

        intval = rb_1.ReadAsArray(px,py,1,1)
        index_1.append(intval[0][0])

        intval = rb_2.ReadAsArray(px,py,1,1)
        index_2.append(intval[0][0])

        intval = rb_3.ReadAsArray(px,py,1,1)
        index_3.append(intval[0][0])

        intval = rb_4.ReadAsArray(px,py,1,1)
        index_4.append(intval[0][0])

        intval = rb_5.ReadAsArray(px,py,1,1)
        index_5.append(intval[0][0])

        intval = rb_6.ReadAsArray(px,py,1,1)
        index_6.append(intval[0][0])

        intval = rb_7.ReadAsArray(px,py,1,1)
        index_7.append(intval[0][0])

        intval = rb_8.ReadAsArray(px,py,1,1)
        index_8.append(intval[0][0])

        intval = rb_9.ReadAsArray(px,py,1,1)
        index_9.append(intval[0][0])

        intval = rb_10.ReadAsArray(px,py,1,1)
        index_10.append(intval[0][0])

        intval = rb_11.ReadAsArray(px,py,1,1)
        index_11.append(intval[0][0])

        intval = rb_12.ReadAsArray(px,py,1,1)
        index_12.append(intval[0][0])

        intval = rb_13.ReadAsArray(px,py,1,1)
        index_13.append(intval[0][0])

        intval = rb_14.ReadAsArray(px,py,1,1)
        index_14.append(intval[0][0])

        intval = rb_15.ReadAsArray(px,py,1,1)
        index_15.append(intval[0][0])

        intval = rb_16.ReadAsArray(px,py,1,1)
        index_16.append(intval[0][0])

        intval = rb_17.ReadAsArray(px,py,1,1)
        index_17.append(intval[0][0])

        intval = rb_18.ReadAsArray(px,py,1,1)
        index_18.append(intval[0][0])

        intval = rb_19.ReadAsArray(px,py,1,1)
        index_19.append(intval[0][0])

        intval = rb_20.ReadAsArray(px,py,1,1)
        index_20.append(intval[0][0])

        kode.append(afgkode)
        markkode.append(mkkode)

    with open(csv_filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['0805_Band_6', '0805_Band_7', '0805_Band_8', '0805_Band_8A', '0805_Band_11', '0805_Band_12',
                             '1104_Band_6', '1104_Band_7', '1104_Band_8', '1104_Band_8A', '1104_Band_11', '1104_Band_12',                                                                                           '1104_Band_6',
                             '1509_Band_6', '1509_Band_7', '1509_Band_8', '1509_Band_8A',
                             '1903_Band_6', '1903_Band_7', '1903_Band_8', '1903_Band_8A', 'AfgKode', 'MarkblokNr'])

        for i in range(0,len(index_1)):
            spamwriter.writerow((index_1[i],index_2[i],index_3[i],index_4[i],index_5[i],index_6[i],
                                 index_7[i], index_8[i], index_9[i], index_10[i], index_11[i], index_12[i],
                                 index_13[i], index_14[i], index_15[i], index_16[i],
                                 index_17[i], index_18[i], index_19[i], index_20[i],
                                 kode[i], markkode[i]))