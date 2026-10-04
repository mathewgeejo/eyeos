pub fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() { return 0.0; }
    let mut v=values.to_vec(); v.sort_by(f64::total_cmp);
    let index=(v.len()-1) as f64*q.clamp(0.0,1.0);
    let low=index.floor() as usize; let high=index.ceil() as usize;
    v[low]+(v[high]-v[low])*(index-low as f64)
}

/// Householder QR on the weighted, ridge-augmented design matrix. Avoids squaring
/// the condition number through normal equations.
pub fn least_squares(mut a: Vec<Vec<f64>>, mut b: Vec<[f64;2]>, n: usize) -> Option<Vec<[f64;2]>> {
    let m=a.len();
    if m<n || b.len()!=m { return None; }
    for k in 0..n {
        let norm=a[k..].iter().map(|r| r[k]*r[k]).sum::<f64>().sqrt();
        if !norm.is_finite() || norm<1e-12 { return None; }
        let alpha=if a[k][k]>=0.0 { -norm } else { norm };
        let mut v: Vec<f64>=a[k..].iter().map(|r| r[k]).collect();
        v[0]-=alpha;
        let vv=v.iter().map(|x| x*x).sum::<f64>();
        if vv<1e-24 { return None; }
        for j in k..n {
            let dot=a[k..].iter().zip(&v).map(|(r,v)| r[j]*v).sum::<f64>()*2.0/vv;
            for (r,v) in a[k..].iter_mut().zip(&v) { r[j]-=dot*v; }
        }
        for axis in 0..2 {
            let dot=b[k..].iter().zip(&v).map(|(r,v)| r[axis]*v).sum::<f64>()*2.0/vv;
            for (r,v) in b[k..].iter_mut().zip(&v) { r[axis]-=dot*v; }
        }
    }
    let mut x=vec![[0.0;2];n];
    for i in (0..n).rev() {
        for axis in 0..2 {
            x[i][axis]=(b[i][axis]-(i+1..n).map(|j| a[i][j]*x[j][axis]).sum::<f64>())/a[i][i];
        }
    }
    x.iter().flatten().all(|v| v.is_finite()).then_some(x)
}
