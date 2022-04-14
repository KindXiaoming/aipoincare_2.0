for i in 1 2 3 4 5
do
    echo "Training Net with seed = $i"
    python train_GHnet.py -config config/harmonic_2d_iso.txt --seed $i
done

<<poincare_one_d_harmonic_damped
for i in 50 60 100 12013 12 41 354 134 5123 42 
do
    echo "Training Net with seed = $i"
    python poincare_2.py -config config/one_d_harmonic_damped.txt --seed $i
done
<<poincare_one_d_harmonic_damped

<<poincare_burgers
for i in 50 60 100 12013 12 41 354 134 5123 42 
do
    echo "Training Net with seed = $i"
    python poincare_2.py -config config/burgers.txt --seed $i
done
<<poincare_burgers

<<poincare_infinite_well_1D
for i in 50 60 100 12013 12 41 354 134 5123 42
do
    echo "Training Net with seed = $i"
    python poincare_2.py -config config/infinite_well_1D.txt --seed $i
done
<<poincare_infinite_well_1D

<<poincare_infinite_well_1D
for i in 552 3267 1324 12 341 547 4567 3141 314 568 11234 13468 5768 9123 
do
    echo "Training Net with seed = $i"
    python poincare_2.py -config config/general_central_force.txt --seed $i
done
<<poincare_infinite_well_1D

<<poincare_kepler_2D
for i in 34 890 70 8796 7890 7897 98 897 7698 98768 876 876 
do
    echo "Training Net with seed = $i"
    python poincare_2.py -config config/kepler_2D.txt --seed $i
done
<<poincare_kepler_2D

<<poincare_anisotropic_harmonic_mesh 
for i in 50 60 100 12013 12 41 354 134 5123 42 || 552 3267 1324 12 341 547 4567 3141 314 568 11234 13468 5768 9123 423 13 13214 1235 87 5689 
do
    echo "Training Net with seed = $i"
    python poincare_2.py -config config/one_d_harmonic.txt --seed $i
done
<<poincare_one_d_harmomic

<<poincare_anisotropic_harmonic_mesh
for i in 50 60 100 12013 12 41 354 134 5123 42 
do
    echo "Training Net with seed = $i"
    python poincare_2.py -config config/anisotropic_harmonic.txt --seed $i
done
<<poincare_anisotropic_harmonic_mesh

<<poincare_isotropic_harmonic_mesh
for i in 42 50 60 100 12013 12 41 354 134 5123
do
    echo "Training Net with seed = $i"
    python poincare_2.py -config config/isotropic_harmonic.txt --seed $i
done
<<poincare_isotropic_harmonic_mesh

<<poincare_isotropic_harmonic_uniform
for i in 42 50 60 100 12013 12 41 354 134 5123
do
    echo "Training Net with seed = $i"
    python poincare_one_d_harmonic_gaussian.py -seed $i
done
<<poincare_isotropic_harmonic_uniform

<<poincare_isotropic_harmonic_uniform
for i in 42 50 60 100 12013 12 41 354 134 5123
do
    echo "Training Net with seed = $i"
    python poincare_isotropic_harmonic_uniform.py -seed $i
done
<<poincare_isotropic_harmonic_uniform
