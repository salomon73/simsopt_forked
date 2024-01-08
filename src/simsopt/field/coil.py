from math import pi
import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import Derivative
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curve import RotatedCurve, Curve
import simsoptpp as sopp


__all__ = ['Coil', 'Current', 'coils_via_symmetries', 'load_coils_from_makegrid_file',
           'apply_symmetries_to_currents', 'apply_symmetries_to_curves',
           'coils_to_makegrid', 'coils_to_focus', 'CoilSet']


class Coil(sopp.Coil, Optimizable):
    """
    A :obj:`Coil` combines a :obj:`~simsopt.geo.curve.Curve` and a
    :obj:`Current` and is used as input for a
    :obj:`~simsopt.field.biotsavart.BiotSavart` field.
    """

    def __init__(self, curve, current):
        self._curve = curve
        self._current = current
        sopp.Coil.__init__(self, curve, current)
        Optimizable.__init__(self, depends_on=[curve, current])

    def vjp(self, v_gamma, v_gammadash, v_current):
        return self.curve.dgamma_by_dcoeff_vjp(v_gamma) \
            + self.curve.dgammadash_by_dcoeff_vjp(v_gammadash) \
            + self.current.vjp(v_current)

    def plot(self, **kwargs):
        """
        Plot the coil's curve. This method is just shorthand for calling
        the :obj:`~simsopt.geo.curve.Curve.plot()` function on the
        underlying Curve. All arguments are passed to
        :obj:`simsopt.geo.curve.Curve.plot()`
        """
        return self.curve.plot(**kwargs)


class CurrentBase(Optimizable):

    def __init__(self, **kwargs):
        Optimizable.__init__(self, **kwargs)

    def __mul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return ScaledCurrent(self, other)

    def __rmul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return ScaledCurrent(self, other)

    def __truediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return ScaledCurrent(self, 1.0/other)

    def __neg__(self):
        return ScaledCurrent(self, -1.)

    def __add__(self, other):
        return CurrentSum(self, other)

    def __sub__(self, other):
        return CurrentSum(self, -other)

    # https://stackoverflow.com/questions/11624955/avoiding-python-sum-default-start-arg-behavior
    def __radd__(self, other):
        # This allows sum() to work (the default start value is zero)
        if other == 0:
            return self
        return self.__add__(other)


class Current(sopp.Current, CurrentBase):
    """
    An optimizable object that wraps around a single scalar degree of
    freedom. It represents the electric current in a coil, or in a set
    of coils that are constrained to use the same current.
    """

    def __init__(self, current, dofs=None, **kwargs):
        sopp.Current.__init__(self, current)
        if dofs is None:
            CurrentBase.__init__(self, external_dof_setter=sopp.Current.set_dofs,
                                 x0=self.get_dofs(), **kwargs)
        else:
            CurrentBase.__init__(self, external_dof_setter=sopp.Current.set_dofs,
                                 dofs=dofs, **kwargs)

    def vjp(self, v_current):
        return Derivative({self: v_current})

    @property
    def current(self):
        return self.get_value()


class ScaledCurrent(sopp.CurrentBase, CurrentBase):
    """
    Scales :mod:`Current` by a factor. To be used for example to flip currents
    for stellarator symmetric coils.
    """

    def __init__(self, current_to_scale, scale, **kwargs):
        self.current_to_scale = current_to_scale
        self.scale = scale
        sopp.CurrentBase.__init__(self)
        CurrentBase.__init__(self, depends_on=[current_to_scale], **kwargs)

    def vjp(self, v_current):
        return self.scale * self.current_to_scale.vjp(v_current)

    def get_value(self):
        return self.scale * self.current_to_scale.get_value()


class CurrentSum(sopp.CurrentBase, CurrentBase):
    """
    Take the sum of two :mod:`Current` objects.
    """

    def __init__(self, current_a, current_b):
        self.current_a = current_a
        self.current_b = current_b
        sopp.CurrentBase.__init__(self)
        CurrentBase.__init__(self, depends_on=[current_a, current_b])

    def vjp(self, v_current):
        return self.current_a.vjp(v_current) + self.current_b.vjp(v_current)

    def get_value(self):
        return self.current_a.get_value() + self.current_b.get_value()


def apply_symmetries_to_curves(base_curves, nfp, stellsym):
    """
    Take a list of ``n`` :mod:`simsopt.geo.curve.Curve`s and return ``n * nfp *
    (1+int(stellsym))`` :mod:`simsopt.geo.curve.Curve` objects obtained by
    applying rotations and flipping corresponding to ``nfp`` fold rotational
    symmetry and optionally stellarator symmetry.
    """
    flip_list = [False, True] if stellsym else [False]
    curves = []
    for k in range(0, nfp):
        for flip in flip_list:
            for i in range(len(base_curves)):
                if k == 0 and not flip:
                    curves.append(base_curves[i])
                else:
                    rotcurve = RotatedCurve(base_curves[i], 2*pi*k/nfp, flip)
                    curves.append(rotcurve)
    return curves


def apply_symmetries_to_currents(base_currents, nfp, stellsym):
    """
    Take a list of ``n`` :mod:`Current`s and return ``n * nfp * (1+int(stellsym))``
    :mod:`Current` objects obtained by copying (for ``nfp`` rotations) and
    sign-flipping (optionally for stellarator symmetry).
    """
    flip_list = [False, True] if stellsym else [False]
    currents = []
    for k in range(0, nfp):
        for flip in flip_list:
            for i in range(len(base_currents)):
                current = ScaledCurrent(base_currents[i], -1.) if flip else base_currents[i]
                currents.append(current)
    return currents


def coils_via_symmetries(curves, currents, nfp, stellsym):
    """
    Take a list of ``n`` curves and return ``n * nfp * (1+int(stellsym))``
    ``Coil`` objects obtained by applying rotations and flipping corresponding
    to ``nfp`` fold rotational symmetry and optionally stellarator symmetry.
    """

    assert len(curves) == len(currents)
    curves = apply_symmetries_to_curves(curves, nfp, stellsym)
    currents = apply_symmetries_to_currents(currents, nfp, stellsym)
    coils = [Coil(curv, curr) for (curv, curr) in zip(curves, currents)]
    return coils


def load_coils_from_makegrid_file(filename, order, ppp=20):
    """
    This function loads a file in MAKEGRID input format containing the Cartesian coordinates 
    and the currents for several coils and returns an array with the corresponding coils. 
    The format is described at
    https://princetonuniversity.github.io/STELLOPT/MAKEGRID

    Args:
        filename: file to load.
        order: maximum mode number in the Fourier expansion.
        ppp: points-per-period: number of quadrature points per period.

    Returns:
        A list of ``Coil`` objects with the Fourier coefficients and currents given by the file.
    """
    with open(filename, 'r') as f:
        all_coils_values = f.read().splitlines()[3:] 

    currents = []
    flag = True
    for j in range(len(all_coils_values)-1):
        vals = all_coils_values[j].split()
        if flag:
            currents.append(float(vals[3]))
            flag = False
        if len(vals) > 4:
            flag = True

    curves = CurveXYZFourier.load_curves_from_makegrid_file(filename, order=order, ppp=ppp)
    coils = [Coil(curves[i], Current(currents[i])) for i in range(len(curves))]

    return coils


def coils_to_makegrid(filename, curves, currents, groups=None, nfp=1, stellsym=False):
    """
    Export a list of Curve objects together with currents in MAKEGRID input format, so they can 
    be used by MAKEGRID and FOCUS. The format is introduced at
    https://princetonuniversity.github.io/STELLOPT/MAKEGRID
    Note that this function does not generate files with MAKEGRID's *output* format.

    Args:
        filename: Name of the file to write.
        curves: A python list of Curve objects.
        currents: Coil current of each curve.
        groups: Coil current group. Coils in the same group will be assembled together. Defaults to None.
        nfp: The number of field periodicity. Defaults to 1.
        stellsym: Whether or not following stellarator symmetry. Defaults to False.
    """

    assert len(curves) == len(currents)
    coils = coils_via_symmetries(curves, currents, nfp, stellsym)
    ncoils = len(coils)
    if groups is None:
        groups = np.arange(ncoils) + 1
    else:
        assert len(groups) == ncoils
        # should be careful. SIMSOPT flips the current, but actually should change coil order
    with open(filename, "w") as wfile:
        wfile.write("periods {:3d} \n".format(nfp)) 
        wfile.write("begin filament \n")
        wfile.write("mirror NIL \n")
        for icoil in range(ncoils):
            x = coils[icoil].curve.gamma()[:, 0]
            y = coils[icoil].curve.gamma()[:, 1]
            z = coils[icoil].curve.gamma()[:, 2]
            for iseg in range(len(x)):  # the last point matches the first one;
                wfile.write(
                    "{:23.15E} {:23.15E} {:23.15E} {:23.15E}\n".format(
                        x[iseg], y[iseg], z[iseg], coils[icoil].current.get_value()
                    )
                )
            wfile.write(
                "{:23.15E} {:23.15E} {:23.15E} {:23.15E} {:} {:10} \n".format(
                    x[0], y[0], z[0], 0.0, groups[icoil], coils[icoil].curve.name
                )
            )
        wfile.write("end \n")
    return


def coils_to_focus(filename, curves, currents, nfp=1, stellsym=False, Ifree=False, Lfree=False):
    """
    Export a list of Curve objects together with currents in FOCUS format, so they can 
    be used by FOCUS. The format is introduced at
    https://princetonuniversity.github.io/FOCUS/rdcoils.pdf
    This routine only works with curves of type CurveXYZFourier,
    not other curve types.

    Args:
        filename: Name of the file to write.
        curves: A python list of CurveXYZFourier objects.
        currents: Coil current of each curve.
        nfp: The number of field periodicity. Defaults to 1.      
        stellsym: Whether or not following stellarator symmetry. Defaults to False.
        Ifree: Flag specifying whether the coil current is free. Defaults to False.
        Lfree: Flag specifying whether the coil geometry is free. Defaults to False.
    """
    from simsopt.geo import CurveLength

    assert len(curves) == len(currents)
    ncoils = len(curves)
    if stellsym:
        symm = 2  # both periodic and symmetric
    elif nfp > 1 and not stellsym:
        symm = 1  # only periodicity
    else:
        symm = 0  # no periodicity or symmetry
    if nfp > 1:
        print('Please note: FOCUS sets Nfp in the plasma file.')
    with open(filename, 'w') as f:
        f.write('# Total number of coils \n')
        f.write('  {:d} \n'.format(ncoils))
        for i in range(ncoils):
            assert isinstance(curves[i], CurveXYZFourier)
            nf = curves[i].order
            xyz = curves[i].full_x.reshape((3, -1))
            xc = xyz[0, ::2]
            xs = np.concatenate(([0.], xyz[0, 1::2]))
            yc = xyz[1, ::2]
            ys = np.concatenate(([0.], xyz[1, 1::2]))
            zc = xyz[2, ::2]
            zs = np.concatenate(([0.], xyz[2, 1::2]))
            length = CurveLength(curves[i]).J()
            nseg = len(curves[i].quadpoints)
            f.write('#------------{:d}----------- \n'.format(i+1))
            f.write('# coil_type  symm  coil_name \n')
            f.write('  {:d}   {:d}  {:} \n'.format(1, symm, curves[i].name))
            f.write('# Nseg current Ifree Length Lfree target_length \n')
            f.write('  {:d} {:23.15E} {:d} {:23.15E} {:d} {:23.15E} \n'.format(nseg, currents[i].get_value(), Ifree, length, Lfree, length))
            f.write('# NFcoil \n')
            f.write('  {:d} \n'.format(nf))
            f.write('# Fourier harmonics for coils ( xc; xs; yc; ys; zc; zs) \n')
            for r in [xc, xs, yc, ys, zc, zs]:  # 6 lines
                for k in range(nf+1):
                    f.write('{:23.15E} '.format(r[k]))
                f.write('\n')
        f.write('\n')
    return


class CoilSet(Optimizable):
    """
    A set of coils as a single optimizable object, and a surface on which 
    it evaluates the field.

    The surfaces' range will be adapted to 'half period' or 'field period'
    if the surface is stellarator symmetric or not.

    Optimization target functions are all available from this class: 
    flux_penalty, length_penalty, cc_distance_penalty, cs_distance_penalty,
    lp_curvature_penalty, meansquared_curvature_penalty, arc_length_variation_penalty
    and total_length.

    These functions can then be used to define a FOCUS-like optimization problem
    (note that the FOCUS algorithm is not a least-squares algorithm) that can be passed
    to scipy.minimize, or the CoilSet can be used as a parent for a SPEC NormalField
    object. 
    """
    #from simsopt.geo import Surface

    def __init__(self, base_coils=None, coils=None, surface=None):  
        from simsopt.field import BiotSavart

        #set the surface, change its's range if necessary
        if surface is None:
            from simsopt.geo import SurfaceRZFourier
            standardsurface = SurfaceRZFourier()
            self._surface = SurfaceRZFourier.from_other_surface(standardsurface, range=SurfaceRZFourier.RANGE_HALF_PERIOD)
        #stellarator symmetric surfaces are more efficiently evaluated on a half-period
        else:
            if surface.stellsym:
                if surface.deduced_range is surface.RANGE_HALF_PERIOD:
                    self._surface = surface
                else:
                    newsurf = surface.to_RZFourier().from_other_surface(surface.to_RZFourier(), range=surface.RANGE_HALF_PERIOD)
                    self._surface = newsurf
            else:
                if surface.deduced_range is surface.RANGE_FIELD_PERIOD:
                    self._surface = surface
                else: 
                    newsurf = surface.to_RZFourier().from_other_surface(surface.to_RZFourier(), range=surface.RANGE_FIELD_PERIOD)
                    self._surface = newsurf

        # set the coils
        if base_coils is not None:
            self.base_coils = base_coils
            if coils is None:
                coils = coils_via_symmetries([coil.curve for coil in base_coils], [coil.current for coil in base_coils], nfp=self._surface.nfp, stellsym=self._surface.stellsym)
            else: 
                self.coils = coils
        else:
            if coils is not None:
                raise ValueError("If base_coils is None, coils must be None as well")
            base_curves = self._cicrclecoils_around_surface(self._surface, nfp=self._surface.nfp, coils_per_period=10)
            base_currents = [Current(1e5) for _ in base_curves]
            base_coils = [Coil(curv, curr) for (curv, curr) in zip(base_curves, base_currents)]
            self.base_coils = base_coils
            self.coils = coils_via_symmetries([coil.curve for coil in base_coils], [coil.current for coil in base_coils], nfp=self._surface.nfp, stellsym=self._surface.stellsym)
        
        self.bs = BiotSavart(self.coils)
        self.bs.set_points(self._surface.gamma().reshape((-1, 3)))
        super().__init__(depends_on=base_coils)

    @classmethod
    def for_surface(cls, surf, coil_current=1e5, coils_per_period=5, nfp=None, current_constraint="fix_all", **kwargs ): 
        """
        Create a CoilSet for a given surface. The coils are created using
        :obj:`create_equally_spaced_curves` with the given parameters.

        Args:
            surf: The surface for which to create the coils
            total_current: the total current in the CoilSet
            coils_per_period: the number of coils per field period
            nfp: The number of field periods.
            current_constraint: "fix_one" or "fix_all" or "free_all" 

        Keyword Args (passed to the create_equally_spaced_curves function)
            coils_per_period: The number of coils per field period
            order: The order of the Fourier expansion
            R0: major radius of a torus on which the initial coils are placed
            R1: The radius of the coils
            factor: if R0 or R1 are None, they are factor times 
                    the first sine/cosine coefficient (factor > 1)
            use_stellsym: Whether to use stellarator symmetry
        """
        if nfp is None:
            nfp = surf.nfp
        base_curves = CoilSet._cicrclecoils_around_surface(surf, coils_per_period=coils_per_period, nfp=nfp, **kwargs)
        base_currents = [Current(coil_current) for _ in base_curves]
        if current_constraint == "fix_one":
            base_currents[0].fix_all()
        elif current_constraint == "fix_all":
            [base_current.fix_all() for base_current in base_currents]
        elif current_constraint == "free_all":
            pass
        else: 
            raise ValueError("current_constraint must be 'fix_one', 'fix_all' or 'free_all'")
        base_coils = [Coil(curv, curr) for (curv, curr) in zip(base_curves, base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=True)
        return cls(base_coils, coils, surf)
    
    @classmethod
    def from_mgrid_file(cls, makegrid_file, surface):
        """
        Create a CoilSet from a MAKEGRID file and a surface.
        """
        coils = load_coils_from_makegrid_file(makegrid_file, order=6, ppp=20)
        return cls(base_coils=coils, coils=coils, surface=surface)

    @classmethod
    def for_spec_equil(cls, spec, coils_per_period=5, current_constraint="fix_all", **kwargs):
        """
        Create a CoilSet for a given SPEC equilibrium. The coils are created using
        :obj:`create_equally_spaced_curves` with the given parameters.

        Args:
            spec: The SPEC object for which to create the coils
            coils_per_period: the number of coils per field period
            nfp: The number of field periods.
            current_constraint: "fix_one" or "fix_all" or "free_all" 

        Keyword Args (passed to the create_equally_spaced_curves function)
            coils_per_period: The number of coils per field period
            order: The order of the Fourier expansion
            R0: major radius of a torus on which the initial coils are placed
            R1: The radius of the coils
            factor: if R0 or R1 are None, they are factor times 
                    the first sine/cosine coefficient (factor > 1)
            use_stellsym: Whether to use stellarator symmetry
        """
        from scipy.constants import mu_0
        nfp = spec.nfp
        use_stellsym = spec.stellsym
        total_current = spec.poloidal_current_amperes
        total_coil_number = coils_per_period * nfp * (1 + int(use_stellsym)) #only coils for half-period
        if spec.freebound: 
            surface = spec.computational_boundary
        else: 
            surface = spec.boundary
        base_curves = CoilSet._cicrclecoils_around_surface(surface, coils_per_period=coils_per_period, nfp=nfp, use_stellsym=use_stellsym, **kwargs)
        base_currents = [Current(total_current/total_coil_number) for _ in base_curves]
        if current_constraint == "fix_one":
            base_currents[0].fix_all()
        elif current_constraint == "fix_all":
            [base_current.fix_all() for base_current in base_currents]
        elif current_constraint == "free_all":
            pass
        else:
            raise ValueError("current_constraint must be 'fix_one', 'fix_all' or 'free_all'")
        base_coils = [Coil(curv, curr) for (curv, curr) in zip(base_curves, base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=True)
        return cls(base_coils, coils, surface)
    
    @property
    def surface(self):
        return self._surface
    
    @surface.setter
    def surface(self, surface: 'Surface'):
        """
        set a new surface for the CoilSet. Change its range if necessary.
        """
        # Changing surface requires changing the BiotSavart object. We also modify its range
        if surface.stellsym: 
            if surface.deduced_range is not surface.RANGE_HALF_PERIOD:
                newsurf = surface.to_RZFourier().from_other_surface(surface.to_RZFourier(), range=surface.RANGE_HALF_PERIOD)
                surface = newsurf
        else:
            if surface.deduced_range is not surface.RANGE_FIELD_PERIOD:
                newsurf = surface.to_RZFourier().from_other_surface(surface.to_RZFourier(), range=surface.RANGE_FIELD_PERIOD)
                surface = newsurf
        self.bs.set_points(surface.gamma().reshape((-1, 3)))
        self._surface = surface
    
    @staticmethod
    def _cicrclecoils_around_surface(surf, nfp=None, coils_per_period=4, order=6, R0=None, R1=None, use_stellsym=None, factor=2.):
        """
        return a set of base curves for a surface using the surfaces properties where possible
        """
        from simsopt.geo import create_equally_spaced_curves
        if use_stellsym is None:
            use_stellsym = surf.stellsym
        if nfp is None:
            nfp = surf.nfp
        elif nfp != surf.nfp:
            raise ValueError("nfp must equal surf.nfp")
        if R0 is None:
            R0 = surf.to_RZFourier().get_rc(0, 0) 
        if R1 is None:
            # take the magnitude of the first-order poloidal Fourier coefficient
            R1 = np.sqrt(surf.to_RZFourier().get_rc(1, 0)**2 + surf.to_RZFourier().get_zs(1, 0)**2) * factor
        return create_equally_spaced_curves(coils_per_period, nfp, stellsym=use_stellsym, R0=R0, R1=R1, order=order)
     
    def get_dof_orders(self):
        """
        get the Fourier order of the degrees of freedom  corresponding to the Fourier
        coefficients describing the coils. 
        Useful for reducing the trust region for the higher order coefficients.
        in optimization. 

        Returns:
            An array with the fourier order of each degree of freedom.
        """
        import re
        dof_names = self.dof_names
        orders = np.ones(self.dof_size)
        # test if coils are CurveXYZFourier:
        if type(self.coils[0].curve) is not CurveXYZFourier:
            raise ValueError("Coils must be of type CurveXYZFourier")
        # run through names, extract order using regular expression
        for name in dof_names:
            if name.startswith("Curve"):
                match = re.search(r'\((\d+)\)', name)
                if match:
                    order = int(match.group(1))
                    if order > 0: # leave zeroth order alone
                        orders[dof_names.index(name)] = order
        return orders

    
    def flux_penalty(self, target=None):
        """
        Return the penalty function for the quadratic flux penalty on 
        the surface
        """
        from simsopt.objectives import SquaredFlux
        return SquaredFlux(self.surface, self.bs, target=target)
    
    def length_penalty(self, TOTAL_LENGTH, f):
        """
        Return a QuadraticPenalty on the total length of the coils 
        if it is larger than TARGET_LENGTH (do not penalize if shorter)
        Args:
            TOTAL_LENGTH: The threshold length above which the penalty is applied
            f: type of penalty, "min", "max" or "identity"
        """
        from simsopt.objectives import QuadraticPenalty
        from simsopt.geo import CurveLength
        # only calculate length of base_coils, others are equal
        coil_multiplication_factor = len(self.coils) / len(self.base_coils)
        # summing optimizables makes new optimizables
        lenth_optimizable = sum(CurveLength(coil.curve) for coil in self.base_coils)*coil_multiplication_factor
        # return the penalty function
        return QuadraticPenalty(lenth_optimizable, TOTAL_LENGTH, f)
    
    def cc_distance_penalty(self, DISTANCE_THRESHOLD): 
        """
        Return a penalty function for the distance between coils
        Args: 
            DISTANCE_THRESHOLD: The threshold distance below which the penalty is applied
        """
        from simsopt.geo import CurveCurveDistance
        curves = [coil.curve for coil in self.coils]
        return CurveCurveDistance(curves, DISTANCE_THRESHOLD, num_basecurves=len(self.base_coils))
    
    def cs_distance_penalty(self, DISTANCE_THRESHOLD):
        """
        Return a penalty function for the distance between coils and the surface
        Args:
            DISTANCE_THRESHOLD: The threshold distance below which the penalty is applied
        """
        from simsopt.geo import CurveSurfaceDistance
        curves = [coil.curve for coil in self.coils]
        return CurveSurfaceDistance(curves, self.surface, DISTANCE_THRESHOLD)

    
    def lp_curvature_penalty(self, CURVATURE_THRESHOLD, p=2):
        """
        Return a penalty function on the curvature of the coils that is 
        the Lp norm of the curvature of the coils. Defaults to L2. 
        Args:
            CURVATURE_THRESHOLD: The threshold curvature above which the penalty is applied
            p: The p in the Lp norm for calculating the penalty
        """
        from simsopt.geo import LpCurveCurvature
        base_curves = [coil.curve for coil in self.base_coils]
        return sum(LpCurveCurvature(curve, p, CURVATURE_THRESHOLD) for curve in base_curves)
    
    def meansquared_curvature_penalty(self):
        """
        Return a penalty function on the mean squared curvature of the coils
        """
        from simsopt.geo import MeanSquaredCurvature
        return sum(MeanSquaredCurvature(coil.curve) for coil in self.base_coils)
    
    def meansquared_curvature_threshold(self, CURVATURE_THRESHOLD):
        """
        Return a penalty function on the mean squared curvature of the coils
        """
        from simsopt.geo import MeanSquaredCurvature
        from simsopt.objectives import QuadraticPenalty
        meansquaredcurvatures = [MeanSquaredCurvature(coil.curve) for coil in self.base_coils]
        return sum(QuadraticPenalty(msc, CURVATURE_THRESHOLD, "max") for msc in meansquaredcurvatures)
    
    def arc_length_variation_penalty(self):
        """
        Return a penalty function on the arc length variation of the coils
        """
        from simsopt.geo import ArclengthVariation
        return sum(ArclengthVariation(coil.curve) for coil in self.base_coils)
    
    def total_length(self):
        """
        Return the total length of the coils
        """
        from simsopt.geo import CurveLength
        multiplicity = len(self.coils) / len(self.base_coils)
        return sum(CurveLength(coil.curve) for coil in self.base_coils)*multiplicity
    
    def to_vtk(self, filename, add_biotsavart=True, close=False):
        """
        Write the CoilSet and its surface to a vtk file
        """
        from simsopt.geo import curves_to_vtk
        curves_to_vtk([coil.curve for coil in self.coils], filename+'_coils', close=close)
        if self.surface is not None:
            if add_biotsavart:
                pointData = {"B_N": np.sum(self.bs.B().reshape((len(self.surface.quadpoints_phi), len(self.surface.quadpoints_theta), 3)) * self.surface.unitnormal(), axis=2)[:, :, None]}
            else: 
                pointData = None
            self.surface.to_vtk(filename+'_surface', extra_data=pointData)


    def plot(self, **kwargs):
        """
        Plot the coil's curve. This method is just shorthand for calling
        the :obj:`~simsopt.geo.Curve/Surface.plot()` function on the
        underlying Curve. All arguments are passed to
        :obj:`simsopt.geo.Curve/Surface.plot()`
        """
        from simsopt.geo import plot, SurfaceRZFourier
        #plot the coils, which are already a list:
        plot(self.coils, **kwargs, show=False)
        #generate a full torus for plotting, plot it:
        SurfaceRZFourier.from_other_surface(self.surface.to_RZFourier(), range=SurfaceRZFourier.RANGE_FULL_TORUS).plot(**kwargs, show=True)

class ReducedCoilSet(CoilSet):
    """
    An Optimizable that replaces a CoilSet but has as degrees of 
    freedom linear combinations of the CoilSet degrees of freedom 

    Single-stage optimization is often accompanied with an explosion
    in the numbers of degrees of freedom needed to accurately represent
    the stellarator, as each coilneeds 6 DOFs per coil per Fourier order. 
    This makes solving the optimization problem harder in two ways: 
    1: the computational cost of finite-difference gradient evaluation is
    linear in the number of DOFs, and 2: the optimization problem can become 
    rank-deficient: changes to your DOFs cancel each other, and the minmum 
    is no longer unique. 

    This class solves these problems by allowing physics-based dimensionality
    reduction of the optimization space using singluar value-decomposition
    of a fast-evalable map. 

    A ReducedCoilSet consists of a Coilset, and the three elements of an SVD of
    the Jacobian defined by a mapping from the coils' DOFs to a physically relevant, 
    fast-evaluating target.  
    """
    def __init__(self, coilset, u_matrix, s_diag, vh_matrix):
        # Reproduce CoilSet's attributes
        self.coilset = coilset
        self.base_coils = coilset.base_coils
        self.coils = coilset.coils
        self._surface = coilset.surface
        self.bs = coilset.bs
        self.bs.set_points(self._surface.gamma().reshape((-1, 3)))
        # Add the SVD matrices
        self._u_matrix = u_matrix
        self._vh_matrix = vh_matrix
        self._coil_x0 = self.coilset.x
        self._s_diag = s_diag
        Optimizable.__init__(self, x0=np.zeros_like(self._s_diag), names=self._make_names(), external_dof_setter=ReducedCoilSet.set_dofs)


    @property
    def surface(self):
        return self.surface

    @surface.setter
    def surface(self, *args, **kwargs):
        raise ValueError("Not supported yet")

    def _make_names(self):
        names = [f"DOFvector{n}" for n in range(len(self.s_diag))]
        return names

    def set_dofs(self, x):
        #check if correct!
        if len(x) != len(self.s_diag):
            raise ValueError("Wrong number of DOFs")
        self.x = x
        self.coilset.x = self._coil_x0 + self._vh_matrix @ (x * self.s_diag)



