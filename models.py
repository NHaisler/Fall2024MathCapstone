import pysindy as ps

def get_sampled_system(points,dt):
    model = ps.SINDy()
    model.fit(points, dt)

    return model