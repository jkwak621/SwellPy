'''
from jakemandel fork
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from peakutils import peak
import crepel
from particle_system_box_xform import ParticleSystem2


class Monodisperse2(ParticleSystem2):
    def __init__(self, N, boxsize_x=None, boxsize_y=None, seed=None):
        """
        Args:
            N (int): The number of particles in the system
            boxsize_x (float): optional. Length of the side(x) of the box
            boxsize_x (float): optional. Length of the side(y) of the box
            seed (int): optional. Seed for initial particle placement randomiztion
        """
        super(Monodisperse2, self).__init__(N, boxsize_x, boxsize_y, seed)
        self._name = "Monodisperse2"
        self.boxsize = np.sqrt(boxsize_x*boxsize_y)
        self.boxsize_x = boxsize_x
        self.boxsize_y = boxsize_y
    
    def equiv_swell(self, area_frac):
        """
        Finds the particle diameter that is equivalent to some area fraction.
        Args:
            area_frac (float): the area fraction of interest
        Returns:
            (float): the equivalent diameter
        """
        af = np.array(area_frac, ndmin=1)
        return 2 * np.sqrt(af * self.boxsize_x*self.boxsize_y / (self.N * np.pi))
    
    def equiv_swell_xform(self, area_frac, scale_x, scale_y):
        """
        Finds the particle diameter that is equivalent to some area fraction, takes in account
        transformation scaling factors.
        ***SHOULD NOT NEED TO BE USED***
        Args:
            area_frac (float): the area fraction of interest
        Returns:
            (float): the equivalent diameter
        """
        af = np.array(area_frac, ndmin=1)
        xform_boxsize_x = (self.boxsize_x*scale_x/scale_y)
        xform_boxsize_y = (self.boxsize_y*scale_y/scale_x)
        swell = 2 * np.sqrt(af * xform_boxsize_x*xform_boxsize_y / (self.N * np.pi))
        return swell
        

    def equiv_area_frac(self, swell):
        """
        Finds the area fraction that is equivalent to some some swell diameter.
        Args:
            swell (float): the particle diameter of interest
        Returns:
            (float) the equivalent area fraction
        """
        d = np.array(swell, ndmin=1)
        return (d / 2)**2 * (self.N * np.pi) / self.boxsize**2

    def _tag(self, swell):
        """ 
        Get the center indices of the particles that overlap at a 
        specific swell
        
        Parameters:
            swell (float): diameter length of the particles
        Returns:
            (np.array): An array object whose elements are pairs of int values that correspond
                the the center indices of overlapping particles
        """

        # Note cKD can retun numpy arrays in query pairs
        # but there is a deallocation bug in the scipy.spatial code
        # converting from a set to an array avoids it
        tree = cKDTree(self.centers, boxsize = self.boxsize)
        pairs = tree.query_pairs(swell)
        pairs = np.array(list(pairs), dtype=np.int64)
        return pairs
    
    def _tag_xform(self, swell, xform_boxsize_x, xform_boxsize_y):
        """ 
        Get the center indices of the particles that overlap at a 
        specific swell. Takes in account the transformation of boxsize.
        
        Parameters:
            swell (float): diameter length of the particles
            xform_boxsize_x (float): X transform boxsize
            xform_boxsize_y (float): Y transform boxsize
        Returns:
            (np.array): An array object whose elements are pairs of int values that correspond
                the the center indices of overlapping particles
        """

        # Note cKD can retun numpy arrays in query pairs
        # but there is a deallocation bug in the scipy.spatial code
        # converting from a set to an array avoids it
        tree = cKDTree(self.centers, boxsize = (xform_boxsize_x, xform_boxsize_y))
        pairs = tree.query_pairs(swell)
        pairs = np.array(list(pairs), dtype=np.int64)
        return pairs
    
    
    def find_angle(self, pairs):
        """
        Finds the kick angles with respect to the first particle.
        """
        theta = []
        for i in pairs:
            x1 = self.centers[i[0]][0] # x-coordinate of first particle
            x2 = self.centers[i[1]][0] # x-coordinate of second particle
            y1 = self.centers[i[0]][1] # y-coordinate of first particle
            y2 = self.centers[i[1]][1] # y-coordinate of second particle
            angle = np.arctan2((y2-y1),(x2-x1))#*(180/np.pi) # angle in degrees
            theta.append(angle)
        return theta
    

    def tag(self, area_frac):
        """
        Finds all tagged particles at some area fraction.
        Args:
            area_frac (float): the area fraction of interest
        Returns:
            (np.array): An array object whose elements are pairs of int values that correspond
                the the center indices of overlapping particles
        """
        swell = self.equiv_swell(area_frac)
        return self._tag(swell)
    
    def repel(self, pairs, area_frac, kick):
        """
        Repels overlapping particles.
        Args:
            pairs (np.array): the pairs of overlapping particles
            area_frac (float): the area fraction of interest
            kick (float): the max kick value the particles are repelled as a percent of the
                inverse diameter
        """
        swell = self.equiv_swell(area_frac)
        self._repel(pairs, swell, kick)


    def train(self, area_frac, kick, cycles=np.inf, noise=0):
        """
        Repeatedly tags and repels overlapping particles for some number of cycles
        
        Args:
            area_frac (float): the area fraction to train on
            kick (float): the maximum distance particles are repelled
            cycles (int): The upper bound on the number of cycles. Defaults to infinite.
        Returns:
            (int) the number of tagging and repelling cycles until no particles overlapped
        """
        
        count = 0
        swell = self.equiv_swell(area_frac)
        pairs = self._tag(swell)
        while (cycles > count and (len(pairs) > 0) ):
            self._repel(pairs, swell, kick)
            self.pos_noise(noise)
            self.wrap()
            pairs = self._tag(swell)
            count += 1
        return count
    
    def xform_boxsize(self, scale_x, scale_y):
        xform_boxsize_x = (self.boxsize_x*scale_x/scale_y)
        xform_boxsize_y = (self.boxsize_y*scale_y/scale_x)
        xform_boxsize = np.sqrt(xform_boxsize_x*xform_boxsize_y)
        return xform_boxsize
    
    def invxform_boxsize(self, scale_x, scale_y):
        xform_boxsize_x = (self.boxsize_x*scale_y/scale_x)
        xform_boxsize_y = (self.boxsize_y*scale_x/scale_y)
        xform_boxsize = xform_boxsize_x*xform_boxsize_y
        return xform_boxsize
    
    def transform_centers(self, scale_x, scale_y):
        for i in self.centers: #Transform centers
                i[0] = i[0]*(scale_x/scale_y)
                i[1] = i[1]*(scale_y/scale_x)
                
    def inv_transform_centers(self, scale_x, scale_y):
        for i in self.centers: #Transform centers back
                i[0] = i[0]*(scale_y/scale_x)
                i[1] = i[1]*(scale_x/scale_y)
    
    def train_xform(self, scale_x, scale_y, area_frac, kick, cycles=np.inf, noise=0):
        """
        Repeatedly transforms system by given amount in given direction, tags particles, transforms back
        to original scale and repels the tagged particles when system what transformed. For some number of cycles
        Args:
            scale_x: scale system in x-direction
            scale_y: scale system in y-direction
            area_frac (float): the area fraction to train on
            kick (float): the maximum distance particles are repelled
            cycles (int): The upper bound on the number of cycles. Defaults to infinite.
        Returns:
            (int) the number of tagging and repelling cycles until no particles overlapped
        """
        count = 0
        swell = self.equiv_swell(area_frac)
        xform_boxsize_x = (self.boxsize_x*scale_x/scale_y)
        xform_boxsize_y = (self.boxsize_y*scale_y/scale_x)
        pairs = self._tag(swell)
        while (cycles > count and (len(pairs) > 0) ):
            for i in self.centers: #Transform centers
                i[0] = i[0]*(scale_x/scale_y)
                i[1] = i[1]*(scale_y/scale_x)
            #self.particle_plot_xformbox(scale_x, scale_y, area_frac, show=True, extend = True, figsize = (7,7), filename=None)
            pairs = self._tag_xform(swell, xform_boxsize_x, xform_boxsize_y) #Tag with box xformed also
            for i in self.centers: #Transform centers back
                i[0] = i[0]*(scale_y/scale_x)
                i[1] = i[1]*(scale_x/scale_y)
            if len(pairs) == 0:
                continue    
            self._repel(pairs, swell, kick)
            self.pos_noise(noise)
            self.wrap()
            count += 1
        return count
    
    def alternate_train(self, scale_x, scale_y, area_frac_x, area_frac_y, kick, cycles=np.inf, noise=0):
        """
        Repeatedly transforms system by given amount in given direction, tags particles, transforms back
        to original scale and repels the tagged particles when system what transformed. For some number of cycles
        
        Will take area_frac_x first then area_frac_y, then continues to alternate.
        Will only scale ONE AXIS AT A TIME
        
        Args:
            scale_x: scale system in x-direction
            scale_y: scale system in y-direction
            area_frac (float): the area fraction to train on
            kick (float): the maximum distance particles are repelled
            cycles (int): The upper bound on the number of cycles. Defaults to infinite.
        Returns:
            (int) the number of tagging and repelling cycles until no particles overlapped
        """
        count = 0
        swell_x = self.equiv_swell(area_frac_x)
        swell_y = self.equiv_swell(area_frac_y)
        xform_boxsize_x_x = (self.boxsize_x*scale_x)   #xform boxsize for area_frac_x transform (Assumes y transform is 1)
        xform_boxsize_y_x = (self.boxsize_y/scale_x)   #xform boxsize for area_frac_x transform
        xform_boxsize_x_y = (self.boxsize_x/scale_y)   #xform boxsize for area_frac_y transform (Assumes x transform is 1)
        xform_boxsize_y_y = (self.boxsize_y*scale_y)   #xform boxsize for area_frac_y transform
        pairs = self._tag(swell_x)
        while (cycles > count and (len(pairs) > 0) ):
            if (count % 2) == 0:
                for i in self.centers: #Transform centers
                    i[0] = i[0]*(scale_x)
                    i[1] = i[1]/scale_x
                pairs = self._tag_xform(swell_x, xform_boxsize_x_x, xform_boxsize_y_x) #Tag with box xformed also
                for i in self.centers: #Transform centers back
                    i[0] = i[0]/scale_x
                    i[1] = i[1]*(scale_x)    
                if len(pairs) == 0:
                    continue    
                self._repel(pairs, swell_x, kick)
                self.pos_noise(noise)
                self.wrap()
                count += 1   
                
            else: #area_frac_y transform
                for i in self.centers: #Transform centers
                    i[0] = i[0]/scale_y
                    i[1] = i[1]*(scale_y)
                pairs = self._tag_xform(swell_y, xform_boxsize_x_y, xform_boxsize_y_y)
                for i in self.centers: #Transform centers back
                    i[0] = i[0]*(scale_y)
                    i[1] = i[1]/scale_y    
                if len(pairs) == 0:
                    continue    
                self._repel(pairs, swell_y, kick)
                self.pos_noise(noise)
                self.wrap()
                count += 1
                
        return count
    
    # def train_xform2(self, axis = ‘x’, ratio):
    #     count = 0
    #     swell = self.equiv_swell(area_frac)
    #     pairs = self._tag(swell)
    #     if (axis == ‘x’):
    #         while (cycles > count and (len(pairs) > 0) ):
    #             for i in self.centers: #Transform
    #                 i[0] = i[0] * ratio
    #             for i in self.centers:
    #                 i[1] = i[1] * (1/ratio)
    #             pairs = self._tag(swell) #Tag
    #             for i in self.centers: #Transform back
    #                 i[0] = i[0] * (1 / ratio)
    #             for i in self.centers:
    #                 i[1] = i[1] * ratio
    #             self._repel(pairs, swell, kick)
    #             self.pos_noise(noise)
    #             self.wrap()
    #             count += 1
    #         return count
    #     else (axis == ‘y’):
    #         while (cycles > count and (len(pairs) > 0) ):
    #             for i in self.centers: #Transform
    #                 i[0] = i[0] * (1/ratio)
    #             for i in self.centers:
    #                 i[1] = i[1] * ratio
    #             pairs = self._tag(swell) #Tag
    #             for i in self.centers: #Transform back
    #                 i[0] = i[0] * ratio
    #             for i in self.centers:
    #                 i[1] = i[1] * (1/ratio)
    #             self._repel(pairs, swell, kick)
    #             self.pos_noise(noise)
    #             self.wrap()
    #             count += 1
    #         return count
                    

    def train_rotxform(self, degrees, scale, area_frac, kick, cycles=np.inf, noise=0):
        """ 
        Rotates system by input degrees, system is scaled by a factor of input scale value, overlapping particles
        are tagged. System is scaled back to original particle 'size.' System is rotated back to original position.
        Particles are then repelled, cycle is complete. Repeat for number of cycles.
        
        Essentially: Training incorporates scaling the system along the axis of the input degrees.
        
        Args:
            degrees: Rotate system by given amount of degrees.
            scale: Scale axis by a specified amount 
            area_frac (float): the area fraction to train on
            kick (float): the maximum distance particles are repelled
            cycles (int): The upper bound on the number of cycles. Defaults to infinite.
        Returns:
            (int) the number of tagging and repelling cycles until no particles overlapped
        """
        count = 0
        swell = self.equiv_swell(area_frac)
        pairs = self._tag(swell)
        theta = np.radians(degrees)
        r = np.array(( (np.cos(theta), -np.sin(theta)),     # Forward Transform Matrix
                      (np.sin(theta),  np.cos(theta)) ))
        while (cycles > count and (len(pairs) > 0) ):
            theta = np.radians(degrees)
            r = np.array(( (np.cos(theta), -np.sin(theta)),
                      (np.sin(theta),  np.cos(theta)) ))
            for i in self.centers:
                [i[0], i[1]] = np.dot(r, [i[0], i[1]])
            for i in self.centers: #scale
                i[1] = i[1]*scale
            for i in self.centers:
                i[0] = i[0]*(1/scale) # Scale perp axis to keep area the same
            #self.particle_plot(area_frac, show=True, extend = True, figsize = (7,7), filename=None)
            #self.wrap()
            pairs = self._tag(swell) # Tag
            r_inv = np.linalg.inv(r)   # Inverse Transform Matrix
            for i in self.centers: # scale
                i[1] = i[1]*(1/scale)
            #self.particle_plot(area_frac, show=True, extend = True, figsize = (7,7), filename=None)
            for i in self.centers:
                i[0] = i[0]*(scale) 
            #self.particle_plot(area_frac, show=True, extend = True, figsize = (7,7), filename=None)
            for i in self.centers:
                [i[0], i[1]] = np.dot(r_inv, [i[0], i[1]])
            #self.particle_plot(area_frac, show=True, extend = True, figsize = (7,7), filename=None)
            self._repel(pairs, swell, kick)
            self.pos_noise(noise)
            self.wrap()
            count += 1
        return count
    

    def particle_plot(self, area_frac, show=True, extend = False, figsize = (7,7), filename=None):
        """
        Show plot of physical particle placement in 2-D box 
        
        Args:
            area_frac (float): The diameter length at which the particles are illustrated
            show (bool): default True. Display the plot after generation
            extend (bool): default False. Show wrap around the periodic boundary.
            figsize ((int,int)): default (7,7). Scales the size of the figure
            filename (string): optional. Destination to save the plot. If None, the figure is not saved. 
        """
        radius = self.equiv_swell(area_frac)/2
        boxsize = self.boxsize
        fig = plt.figure(figsize = figsize)
        plt.axis('off')
        ax = plt.gca()
        for pair in self.centers:
            ax.add_artist(Circle(xy=(pair), radius = radius))
            if (extend):
                ax.add_artist(Circle(xy=(pair) + [0, boxsize], radius = radius, alpha=0.5))
                ax.add_artist(Circle(xy=(pair) + [boxsize, 0], radius = radius, alpha=0.5))
                ax.add_artist(Circle(xy=(pair) + [boxsize, boxsize], radius = radius, alpha=0.5))
        if (extend):
            plt.xlim(0, 2*boxsize)
            plt.ylim(0, 2*boxsize)
            plt.plot([0, boxsize*2], [boxsize, boxsize], ls = ':', color = '#333333')
            plt.plot([boxsize, boxsize], [0, boxsize*2], ls = ':', color = '#333333')

        else:
            plt.xlim(0, boxsize)
            plt.ylim(0, boxsize)
        fig.tight_layout()
        if filename != None:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()
        
    def particle_plot_xformbox(self, scale_x, scale_y, area_frac, show=True, extend = False, figsize = (7,7), filename=None):
        """
        Show plot of physical particle placement in 2-D box 
        
        Args:
            area_frac (float): The diameter length at which the particles are illustrated
            show (bool): default True. Display the plot after generation
            extend (bool): default False. Show wrap around the periodic boundary.
            figsize ((int,int)): default (7,7). Scales the size of the figure
            filename (string): optional. Destination to save the plot. If None, the figure is not saved. 
        """
        radius = self.equiv_swell_xform(area_frac, scale_x, scale_y)/2
        xform_boxsize_x = (self.boxsize_x*(scale_x/scale_y))
        xform_boxsize_y = (self.boxsize_y*(scale_y/scale_x))
        boxsize = (self.boxsize)
        print(boxsize)
        print(xform_boxsize_x)
        print(xform_boxsize_y)
        fig = plt.figure(figsize = figsize)
        plt.axis('off')
        ax = plt.gca()
        for pair in self.centers:
            ax.add_artist(Circle(xy=(pair), radius = radius))
            if (extend):
                ax.add_artist(Circle(xy=(pair) + [0, xform_boxsize_y], radius = radius, alpha=0.5))
                ax.add_artist(Circle(xy=(pair) + [xform_boxsize_x, 0], radius = radius, alpha=0.5))
                ax.add_artist(Circle(xy=(pair) + [xform_boxsize_x, xform_boxsize_y], radius = radius, alpha=0.5))
        if (extend):
            plt.xlim(0, 2*xform_boxsize_x)
            plt.ylim(0, 2*xform_boxsize_y)
            plt.plot([0, xform_boxsize_y*2], [xform_boxsize_y, xform_boxsize_y], ls = ':', color = '#333333')
            plt.plot([xform_boxsize_x, xform_boxsize_x], [0, xform_boxsize_y*2], ls = ':', color = '#333333')

        else:
            plt.xlim(0, xform_boxsize_x)
            plt.ylim(0, xform_boxsize_y)
        fig.tight_layout()
        if filename != None:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()

    def _tag_count(self, swells):
        """
        Returns the number of tagged pairs at a specific area fraction
        
        Args:
            swell (float): swollen diameter length of the particles
        Returns:
            (float): The fraction of overlapping particles
        """
        i = 0
        tagged = np.zeros(swells.size)
        while i < swells.size:
            temp = self._tag(swells[i])
            tagged[i] = np.unique(temp).size/ self.N
            i += 1
        return tagged
    
    def tag_count(self, area_frac):
        """
        Returns the number of tagged pairs at a specific area fraction
        
        Args:
            area_frac (float): area fraction of the particles
        Returns:
            (float): The fraction of overlapping particles
        """
        swells = self.equiv_swell(area_frac)
        return self._tag_count(swells)

    def _extend_domain(self, domain):
        """
        Inserts a value at the beginning of the domain equal to the separation between the first
        two values, and a value at the end of the array determined by the separation of the last
        two values
        Args:
            domain (np.array): array to extend
        Return:
            (np.array) extended domain array
        """
        first = 2 * domain[0] - domain[1]
        if (first < 0):
            first = 0
        last = 2 * domain[-1] - domain[-2]
        domain_extend = np.insert(domain, 0, first)
        domain_extend = np.append(domain_extend, last)
        return domain_extend

    
    def tag_rate(self, area_frac):
        """
        Returns the rate at which the fraction of particles overlap over a range of area fractions.
        This is the same as measuring the fraction tagged at two area fractions and dividing by the 
        difference of the area fractions. 
        
        Args:
            area_frac (np.array): array fractions to calculate tag rate at
        Returns:
            (np.array): The rate of the fraction of tagged particles at area fraction in the input array
        """
        af_extended = self._extend_domain(area_frac)
        tagged = self.tag_count(af_extended)
        rate = (tagged[2:] - tagged[:-2])
        return rate

    def tag_curve(self, area_frac):
        """
        Returns the curvature at which the fraction of particles overlap over a range of area fractions.
        This is the same as measuring the rate at two area fractions and dividing by the difference
        of the area fractions. 
        
        Args:
            area_frac (np.array): array fractions to calculate the tag curvature at
        Returns:
            (np.array): The curvature of the fraction of tagged particles at area fraction in the input array
        """
        af_extended = self._extend_domain(area_frac)
        rate = self.tag_rate(af_extended)
        curve = (rate[2:] - rate[:-2])
        return curve

    def tag_plot(self, area_frac, mode='count', show=True, filename=None):
        """
        Generates a plot of the tag count, rate, or curvature
        Args:
            area_frac (np.array): list of the area fractions to use in the plot
            mode ("count"|"rate"|"curve"): which information you want to plot. Defaults to "count".
            show (bool): default True. Whether or not to show the plot
            filename (string): default None. Filename to save the plot as. If filename=None, the plot is not saved.
        """
        if (mode == 'curve'):
            plt.ylabel('Curve')
            func = self.tag_curve
        elif (mode == 'rate'):
            plt.ylabel('Rate')
            func = self.tag_rate
        else:
            plt.ylabel('Count')
            func = self.tag_count
        data = func(area_frac) 
        plt.plot(area_frac, data)
        plt.xlabel("Area Fraction")
        if filename:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()

    def detect_memory(self, start, end, incr):
        """
        Tests the number of tagged particles over a range of area fractions, and 
        returns a list of area fractions where memories are detected. 
        
        Args:
            start (float): The first area fraction in the detection
            end (float): The last area fraction in the detection
            incr (float): The increment between test swells. Determines accuracy of the memory detection. 
        Returns:
            (np.array): list of swells where a memory is located
        """
        area_frac = np.arange(start, end, incr)
        curve = self.tag_curve(area_frac)
        zeros = np.zeros(curve.shape)
        pos = np.choose(curve < 0, [curve, zeros])
        neg = np.choose(curve > 0, [curve, zeros])
        indices = peak.indexes(pos, 0.5, incr)
        nindices = peak.indexes(-neg, 0.5, incr)
        matches = []
        for i in indices:
            for j in nindices:
                desc = True
                if (i < j):
                    for k in range(i,j):
                        if (curve[k] < curve[k+1]):
                            desc = False
                    if (desc):
                        matches.append(i)
        return area_frac[matches]
    
    
    def _tag_count_xform(self, swells, scale_x, scale_y):
        """
        Returns the number of tagged pairs at a specific area fraction
        
        Args:
            swell (float): swollen diameter length of the particles
        Returns:
            (float): The fraction of overlapping particles
        """
        i = 0
        tagged = np.zeros(swells.size)
        xform_boxsize_x = (self.boxsize_x*(scale_x/scale_y))
        xform_boxsize_y = (self.boxsize_y*(scale_y/scale_x))
        while i < swells.size:
            temp = self._tag_xform(swells[i], xform_boxsize_x, xform_boxsize_y)
            tagged[i] = np.unique(temp).size/ self.N
            i += 1
        return tagged
    
    def tag_count_xform(self, area_frac, scale_x, scale_y):
        """
        Returns the number of tagged pairs at a specific area fraction
        
        Args:
            area_frac (float): area fraction of the particles
        Returns:
            (float): The fraction of overlapping particles
        """
        swells = self.equiv_swell(area_frac)
        return self._tag_count_xform(swells, scale_x, scale_y)

    '''
    Use original extend domain function
    '''
    # def _extend_domain_xform(self, domain):
    #     """
    #     Inserts a value at the beginning of the domain equal to the separation between the first
    #     two values, and a value at the end of the array determined by the separation of the last
    #     two values

    #     Args:
    #         domain (np.array): array to extend
    #     Return:
    #         (np.array) extended domain array
    #     """
    #     first = 2 * domain[0] - domain[1]
    #     if (first < 0):
    #         first = 0
    #     last = 2 * domain[-1] - domain[-2]
    #     domain_extend = np.insert(domain, 0, first)
    #     domain_extend = np.append(domain_extend, last)
    #     return domain_extend

    
    def tag_rate_xform(self, area_frac, scale_x, scale_y):
        """
        Returns the rate at which the fraction of particles overlap over a range of area fractions.
        This is the same as measuring the fraction tagged at two area fractions and dividing by the 
        difference of the area fractions. 
        
        Args:
            area_frac (np.array): array fractions to calculate tag rate at
        Returns:
            (np.array): The rate of the fraction of tagged particles at area fraction in the input array
        """
        af_extended = self._extend_domain(area_frac)
        tagged = self.tag_count_xform(af_extended, scale_x, scale_y)
        rate = (tagged[2:] - tagged[:-2])
        return rate

    def tag_curve_xform(self, area_frac, scale_x, scale_y):
        """
        Returns the curvature at which the fraction of particles overlap over a range of area fractions.
        This is the same as measuring the rate at two area fractions and dividing by the difference
        of the area fractions. 
        
        Args:
            area_frac (np.array): array fractions to calculate the tag curvature at
        Returns:
            (np.array): The curvature of the fraction of tagged particles at area fraction in the input array
        """
        af_extended = self._extend_domain(area_frac)
        rate = self.tag_rate_xform(af_extended, scale_x, scale_y)
        curve = (rate[2:] - rate[:-2])
        return curve

    def tag_plot_xform(self, scale_x, scale_y, area_frac, mode='count', show=True, filename=None):
        """
        Generates a plot of the tag count, rate, or curvature
        Args:
            scale_x (float, optional):
            scale_y (float, optional):
            area_frac (np.array): list of the area fractions to use in the plot
            mode ("count"|"rate"|"curve"): which information you want to plot. Defaults to "count".
            show (bool): default True. Whether or not to show the plot
            filename (string): default None. Filename to save the plot as. If filename=None, the plot is not saved.
        """
        for i in self.centers: #Transform centers along readout axis
                i[0] = i[0]*(scale_x/scale_y)
                i[1] = i[1]*(scale_y/scale_x)
        if (mode == 'curve'):
            plt.ylabel('Curve')
            func = self.tag_curve_xform
        elif (mode == 'rate'):
            plt.ylabel('Rate')
            func = self.tag_rate_xform
        else:
            plt.ylabel('Count')
            func = self.tag_count_xform
        data = func(area_frac, scale_x, scale_y) 
        plt.plot(area_frac, data)
        plt.xlabel("Area Fraction")
        for i in self.centers: #Transform centers back
                i[0] = i[0]*(scale_y/scale_x)
                i[1] = i[1]*(scale_x/scale_y)
        if filename:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()
    
    def tag_overlay_plot2(self, area_frac, scale_x, scale_y, mode='count', show=True, filename=None):
        '''
        area_frac is area fraction array
        
        Parameters
        ----------
        area_frac : TYPE
            DESCRIPTION.
        scale_x : TYPE
            Scaling factor on x-axis
        scale_y : TYPE
            Scaling factor on y-axis
        Returns
        -------
        None.
        '''
        if (mode == 'curve'):
            plt.ylabel('Curve')
            funcI = self.tag_curve_xform
            dataI = funcI(area_frac, 1, 1)
            #Transform for x-axis readout
            self.transform_centers(scale_x, scale_y)
            funcX = self.tag_curve_xform
            dataX = funcX(area_frac, scale_x, scale_y)
            self.inv_transform_centers(scale_x, scale_y) #transform centers back
            #Transform for y-axis readout
            self.transform_centers(scale_y, scale_x)
            funcY = self.tag_curve_xform
            dataY = funcY(area_frac, scale_y, scale_x)
            self.inv_transform_centers(scale_y, scale_x) #transform centers back
        elif (mode == 'rate'):
            plt.ylabel('Rate')
            funcI = self.tag_rate_xform
            dataI = funcI(area_frac, 1, 1)
            #Transform for x-axis readout
            self.transform_centers(scale_x, scale_y)
            funcX = self.tag_rate_xform
            dataX = funcX(area_frac, scale_x, scale_y)
            self.inv_transform_centers(scale_x, scale_y) #transform centers back
            #Transform for y-axis readout
            self.transform_centers(scale_y, scale_x)
            funcY = self.tag_rate_xform
            dataY = funcY(area_frac, scale_y, scale_x)
            self.inv_transform_centers(scale_y, scale_x) #transform centers back
        else:
            plt.ylabel('Count')
            funcI = self.tag_count_xform
            dataI = funcI(area_frac, 1, 1)
            #Transform for x-axis readout
            self.transform_centers(scale_x, scale_y)
            funcX = self.tag_count_xform
            dataX = funcX(area_frac, scale_x, scale_y)
            self.inv_transform_centers(scale_x, scale_y) #transform centers back
            #Transform for y-axis readout
            self.transform_centers(scale_y, scale_x)
            funcY = self.tag_count_xform
            dataY = funcY(area_frac, scale_y, scale_x)
            self.inv_transform_centers(scale_y, scale_x) #transform centers back
        plt.plot(area_frac, dataI)
        plt.plot(area_frac, dataX)
        plt.plot(area_frac, dataY)
        plt.xlabel('Area Fraction')
        plt.legend(['Isotropic memory', 'X-axis memory', 'Y-axis memory'])
        if show == True:
            plt.show()
        plt.close()
            
            


    # def tag_overlay_plot(self, area_frac, scale_x, scale_y, mode='count', show=True):
    #     if (mode == 'curve'):
    #         plt.ylabel('Curve')
    #         funcI = self.tag_curve_xform
    #         dataI = funcI(area_frac, 1, 1)
    #         for i in self.centers: # scaling for memory readout by x-axis
    #             i[0] = i[0]*(scale_x/scale_y)
    #             i[1] = i[1]*(scale_y/scale_x)
    #         funcX = self.tag_curve_xform
    #         dataX = funcX(area_frac, scale_x, scale_y)
    #         for i in self.centers: #Transform centers back and Switches ratios for memory readout by y-axis
    #             i[0] = i[0]((scale_y/scale_x)*2)
    #             i[1] = i[1]((scale_x/scale_y)*2)
    #         funcY = self.tag_curve_xform
    #         dataY = funcY(area_frac, scale_x, scale_y)
    #         for i in self.centers: # Transform centers back
    #             i[0] = i[0]*(scale_x/scale_y)
    #             i[1] = i[1]*(scale_y/scale_x)
    #     elif (mode == 'rate'):
    #         plt.ylabel('Rate')
    #         funcI = self.tag_rate_xform
    #         dataI = funcI(area_frac, 1, 1)
    #         for i in self.centers: # scaling for memory readout by x-axis
    #             i[0] = i[0]*(scale_x/scale_y)
    #             i[1] = i[1]*(scale_y/scale_x)
    #         funcX = self.tag_rate_xform
    #         dataX = funcX(area_frac, scale_x, scale_y)
    #         for i in self.centers: #Transform centers back
    #             i[0] = i[0]*(scale_y/scale_x)
    #             i[1] = i[1]*(scale_x/scale_y)
    #         for i in self.centers: # Switches ratios for memory readout by y-axis
    #             i[0] = i[0]*(scale_y/scale_x)
    #             i[1] = i[1]*(scale_x/scale_y)
    #         funcY = self.tag_rate_xform
    #         dataY = funcY(area_frac, scale_x, scale_y)
    #         for i in self.centers: # Transform centers back
    #             i[0] = i[0]*(scale_x/scale_y)
    #             i[1] = i[1]*(scale_y/scale_x)
    #     else:
    #         plt.ylabel('Count')
    #         funcI = self.tag_count_xform
    #         dataI = funcI(area_frac, 1, 1)
    #         for i in self.centers: # scaling for memory readout by x-axis
    #             i[0] = i[0]*(scale_x/scale_y)
    #             i[1] = i[1]*(scale_y/scale_x)
    #         funcX = self.tag_count_xform
    #         dataX = funcX(area_frac, scale_x, scale_y)
    #         for i in self.centers: #Transform centers back
    #             i[0] = i[0]*(scale_y/scale_x)
    #             i[1] = i[1]*(scale_x/scale_y)
    #         for i in self.centers: # Switches ratios for memory readout by y-axis
    #             i[0] = i[0]*(scale_y/scale_x)
    #             i[1] = i[1]*(scale_x/scale_y)
    #         funcY = self.tag_count_xform
    #         dataY = funcY(area_frac, scale_x, scale_y)
    #         for i in self.centers: # Transform centers back
    #             i[0] = i[0]*(scale_x/scale_y)
    #             i[1] = i[1]*(scale_y/scale_x)
    #     plt.plot(area_frac, dataI)
    #     plt.plot(area_frac, dataX)
    #     plt.plot(area_frac, dataY)
    #     plt.xlabel('Area Fraction')
    #     plt.legend(['Isotropic memory', 'X-axis memory', 'Y-axis memory'])
    #     if show == True:
    #         plt.show()
    #     plt.close()
           
        

    
    def detect_memory_xform(self, start, end, incr, scale_x = 1,scale_y = 1):
        """
        Tests the number of tagged particles over a range of area fractions, and 
        returns a list of area fractions where memories are detected. 
        
        Args:
            start (float): The first area fraction in the detection
            end (float): The last area fraction in the detection
            incr (float): The increment between test swells. Determines accuracy of the memory detection. 
        Returns:
            (np.array): list of swells where a memory is located
        """
        for i in self.centers: #Transform centers along readout axis
                i[0] = i[0]*(scale_x/scale_y)
                i[1] = i[1]*(scale_y/scale_x)
        area_frac = np.arange(start, end, incr)
        curve = self.tag_curve_xform(area_frac, scale_x, scale_y)
        zeros = np.zeros(curve.shape)
        pos = np.choose(curve < 0, [curve, zeros])
        neg = np.choose(curve > 0, [curve, zeros])
        indices = peak.indexes(pos, 0.5, incr)
        nindices = peak.indexes(-neg, 0.5, incr)
        matches = []
        for i in indices:
            for j in nindices:
                desc = True
                if (i < j):
                    for k in range(i,j):
                        if (curve[k] < curve[k+1]):
                            desc = False
                    if (desc):
                        matches.append(i)
        for i in self.centers: #Transform centers back
                i[0] = i[0]*(scale_y/scale_x)
                i[1] = i[1]*(scale_x/scale_y)
        return area_frac[matches]
    
    # def find_axis(self,theta):
        '''
        Returns what axis is being read off of. NOT the axis the memory was written
        '''
    #         axis = 'Isotropic'
    #         axis = 'X'
    #         axis = 'Y'
    #     return axis
    
    def find_written_memory(self, N, B, axis, memory, scale, kick=0):
        '''
        Can replace input memory for output of detect memory function
            -Need to add how to account for kick error
        '''
        #check isotropic reading
        # 1) kicks
        # 2) map back
        if axis == 'Isotropic':
            written_memory = memory/(scale**2)
        #if iso fails check x or y
        elif axis == 'x':
            written_memory = memory/(scale**4)
        elif axis == 'y':
            written_memory = memory/(scale**4)
        else:
            written_memory = 0
            print('ERROR: AXIS NOT FOUND')
        return print('Written Memory:','\n','Axis:', axis,'\n', 'Area Fraction:', written_memory)
       
