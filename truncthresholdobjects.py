__author__ = "Nuno Lages"
__email__ = "lages@uthscsa.edu"


import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import scipy
import scipy.stats as stats
import scipy.io as sio
# import wx

# import cPickle as pickle


class TruncThresholdObjects(cpm.CPModule):

    variable_revision_number = 1
    module_name = "TruncThresholdObjects"
    category = "Image Processing"

    def create_settings(self):

        self.input_image_name = cps.ImageNameSubscriber(
            # The text to the left of the edit box
            "Input image name:",
            # HTML help that gets displayed when the user presses the
            # help button to the right of the edit box
            doc = """This is the image that the module operates on. You can
            choose any image that is made available by a prior module.
            <br>
            <b>ImageTemplate</b> will do something to this image.
            """
        )

        self.output_image_name = cps.ImageNameProvider(
            "Output image name:",
            # The second parameter holds a suggested name for the image.
            "OutputImage",
            doc="""This is the image resulting from the operation."""
        )

        self.input_objects_name = cps.ObjectNameSubscriber(
            # The text to the left of the edit box
            "Input objects name:",
            # HTML help that gets displayed when the user presses the
            # help button to the right of the edit box
            doc = """This is the objects that the module operates on. You can
            choose any objects that is made available by a prior module.
            <br>
            <b>TruncThresholdObjects</b> will do something to this objects.
            """
        )

        self.center = cps.Choice(
            "Center choice:",
            # The choice takes a list of possibilities. The first one
            # is the default - the one the user will typically choose.
            ['median', 'average'],
            doc="""Choose what to use as estimate of the mean of the
                truncated normal distribution."""
        )

        self.scale_r = cps.Float(
            "Truncated normal parameter red channel:",
            # The default value
            3000.0,
            doc=""""""
        )

        self.scale_g = cps.Float(
            "Truncated normal parameter green channel:",
            # The default value
            3000.0,
            doc=""""""
        )

        self.scale_b = cps.Float(
            "Truncated normal parameter blue channel:",
            # The default value
            3000.0,
            doc=""""""
        )

        self.percentile_r = cps.Float(
            "Percentile red channel:",
            # The default value
            0.01,
            doc=""""""
        )

        self.percentile_g = cps.Float(
            "Percentile green channel:",
            # The default value
            0.01,
            doc=""""""
        )

        self.percentile_b = cps.Float(
            "Percentile blue channel:",
            # The default value
            0.0,
            doc=""""""
        )

    def settings(self):
        return [self.input_image_name,
                self.output_image_name,
                self.input_objects_name,
                self.center,
                self.scale_r,
                self.scale_g,
                self.scale_b,
                self.percentile_r,
                self.percentile_g,
                self.percentile_b]

    def run(self, workspace):

        diagnostics = dict()

        cent = self.center.get_value()

        input_objects_name = self.input_objects_name.value
        object_set = workspace.object_set
        assert isinstance(object_set, cpo.ObjectSet)

        input_image_name = self.input_image_name.value
        image_set = workspace.image_set
        assert isinstance(image_set, cpi.ImageSet)
        output_image_name = self.output_image_name.value

        input_image = image_set.get_image(input_image_name)# must_be_rgb=True)
        pixels = input_image.pixel_data
        diagnostics['pixels'] = pixels

        input_objects = object_set.get_objects(input_objects_name)

        mask = input_objects.get_segmented()

        new_im = scipy.zeros(shape=pixels.shape)

        diagnostics['new_im'] = list()
        diagnostics['nucleus_processed'] = list()
        diagnostics['nucleus_pixels'] = list()
        diagnostics['ci'] = list()

        for x in range(1, mask.max()+1):

            nucleus_map = mask == x

            nucleus_pixels = scipy.zeros(shape=pixels.shape)
            for i, j, k in scipy.nditer([pixels,
                                         nucleus_map[:, :, scipy.newaxis],
                                         nucleus_pixels],
                                        op_flags=[['readonly'],
                                                  ['readonly'],
                                                  ['readwrite']]):
                for a, b, c in scipy.nditer([i, j, k],
                                            op_flags=[['readonly'],
                                                      ['readonly'],
                                                      ['readwrite']]):
                    if b:
                        c[...] = scipy.copy(a)

            diagnostics['nucleus_pixels'].append(nucleus_pixels)

            nucleus_pixels_t = scipy.transpose(nucleus_pixels)

            nucleus_ci_r = get_ci(nucleus_pixels_t[0],
                                  percentile=self.percentile_r.get_value(),
                                  center=cent,
                                  mod=self.scale_r.get_value())

            nucleus_ci_g = get_ci(nucleus_pixels_t[1],
                                  percentile=self.percentile_g.get_value(),
                                  center=cent,
                                  mod=self.scale_g.get_value())

            nucleus_ci_b = get_ci(nucleus_pixels_t[2],
                                  percentile=self.percentile_b.get_value(),
                                  center=cent,
                                  mod=self.scale_b.get_value())

            diagnostics['ci'].append((nucleus_ci_r, nucleus_ci_g,
                                      nucleus_ci_b))

            nucleus_processed = update_image(nucleus_pixels,
                                             nucleus_ci_r,
                                             nucleus_ci_g,
                                             nucleus_ci_b)

            diagnostics['nucleus_processed'].append(nucleus_processed)

            new_im = new_im + nucleus_processed

            diagnostics['new_im'].append(new_im)

            # with open('/Users/lages/Documents/sauceda/pictures_processed/diagnostics'
            #           '.p', 'wb') as f:
            #     pickle.dump(diagnostics, f)

            from os.path import expanduser
            home = expanduser("~")

            # with open(home + '/ci_values.txt', 'wb') as f:
            #     writeListsToLines(diagnostics['ci'], f)

            sio.savemat(home + '/diagnostics.mat', diagnostics)

        output_image = cpi.Image(new_im, parent_image=input_image)
        image_set.add(output_image_name, output_image)

    def is_interactive(self):
        return False


def var_truncNormal(a, b, mu, sigma, data, mod=3000.0):

    x1 = (a - mu)/sigma * stats.norm.pdf(a, mu, sigma)
    x2 = (b - mu)/sigma * stats.norm.pdf(b, mu, sigma)

    cx = stats.norm.cdf(b, mu, sigma) - stats.norm.cdf(a, mu, sigma)

    yhat = stats.tvar(data, limits=[mu-mod, mu+mod], inclusive=(False, False))
    sigma2 = yhat/((1+(x1-x2)/cx - ((x1-x2)/cx)**2))
    sigma = scipy.sqrt(sigma2)

    return sigma


def update_image(original_im, ci_red, ci_green, ci_blue):

    # diagnostics = dict()

    original_im = scipy.transpose(original_im)
    # diagnostics['original_im'] = original_im
    # diagnostics['ci_red'] = ci_red
    # diagnostics['ci_green'] = ci_green
    # diagnostics['ci_blue'] = ci_blue

    new_r = scipy.multiply(original_im[0], original_im[0] > ci_red)

    new_g = scipy.multiply(original_im[1], original_im[1] > ci_green)

    new_b = scipy.multiply(original_im[2], original_im[2] > ci_blue)

    new_im = (new_r, new_g, new_b)

    new_im = scipy.transpose(new_im)
    # diagnostics['new_im'] = new_im

    # with open('/Users/lages/Documents/sauceda/pictures_processed/diagnostics'
    #           '.p', 'wb') as f:
    #     pickle.dump(diagnostics, f)

    return new_im


def get_ci(im_data, center='median', mod=3000.0, percentile=0.01):

    flattened = scipy.concatenate(im_data)
    flattened = flattened[scipy.nonzero(flattened)]

    if center == 'median':
        mu = scipy.median(flattened)
    elif center == 'mean':
        mu = scipy.average(flattened)

    sigma = stats.tstd(flattened)

    mod == scipy.float_(mod)
    sigma = var_truncNormal(mu - mod, mu + mod, mu, sigma, flattened, mod=mod)

    ci = 2 * mu - stats.norm.ppf(percentile, mu, sigma)

    return ci


def writeListsToLines(l_list, f_obj):

    '''Takes a list of lists, l_list and writes each of them in a different line
    in a tab-delimited file with name f_name.'''

    for l in l_list:
        for s in range(len(l) - 1):
            f_obj.write(str(l[s]) + '\t')
        f_obj.write(str(l[len(l) - 1]))
        f_obj.write('\n')