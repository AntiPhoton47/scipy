from libc.math cimport log, fabs, expm1, log1p, isnan, NAN, INFINITY, isinf, fmin

cdef inline double entr(double x, double b) nogil:
    if b > 1:
        if isnan(x):
            return x
        elif x > 0:
            return -x * log(x)/log(b)
        elif x == 0:
            return 0
        else:
            return -INFINITY
    else:
        return NAN
    
cdef inline double ren_entr(double x, double a, double b) nogil:
    if a < 0:
        return NAN
    elif a == 1:
        return entr(x, b)
    elif isinf(a):
        return fmin(self_info(x, b))
    elif a == 0:
        return log(sizeof(x)/sizeof(x[0]))
    else:
        if isnan(x):
            return x
        elif x > 0:
            return 1/(1-a) * log(x**a)
        elif x == 0 and a :
            return 0
        else:
            return -INFINITY
        
cdef inline double self_info(double x, double b) nogil:
    if b > 1:
        if isnan(x):
            return x
        elif x > 0:
            return -log(x)/log(b)
        elif x == 0:
            return -INFINITY
        else:
            return NAN
    else:
        return NAN

cdef inline double kl_div(double x, double y) nogil:
    if isnan(x) or isnan(y):
        return NAN
    elif x > 0 and y > 0:
        return x * log(x / y) - x + y
    elif x == 0 and y >= 0:
        return y
    else:
        return INFINITY

cdef inline double rel_entr(double x, double y) nogil:
    if isnan(x) or isnan(y):
        return NAN
    elif x > 0 and y > 0:
        return x * log(x / y)
    elif x == 0 and y >= 0:
        return 0
    else:
        return INFINITY

cdef inline double huber(double delta, double r) nogil:
    if delta < 0:
        return INFINITY
    elif fabs(r) <= delta:
        return 0.5 * r * r;
    else:
        return delta * (fabs(r) - 0.5 * delta);

cdef inline double pseudo_huber(double delta, double r) nogil:
    cdef double u, v
    if delta < 0:
        return INFINITY
    elif delta == 0 or r == 0:
        return 0
    else:
        u = delta
        v = r / delta
        # The formula is u*u*(sqrt(1 + v*v) - 1), but to maintain
        # precision with small v, we use
        #   sqrt(1 + v*v) - 1  =  exp(0.5*log(1 + v*v)) - 1
        #                      =  expm1(0.5*log1p(v*v))
        return u*u*expm1(0.5*log1p(v*v))
