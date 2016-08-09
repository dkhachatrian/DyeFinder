// ImageJ macro for saving anisotropy data of an image, as Text Images


// Is to be used via commandline-like interface

path = getArgument(); //pathname

SENTINEL_FILE = "batched.txt"

//args = split(string, " ") //dep_dir, file_name

function perform_orientation_analysis(fpath)
{
	
setBatchMode(true);
	open(fpath);
	fname = File.getName(fpath)
	parent = File.getParent(fpath)
	run("8-bit");
	//run("OrientationJ Analysis");
	run("OrientationJ Analysis", "log=0.0 tensor=1.0 gradient=0 energy=on orientation=on coherency=on harris-index=on s-distribution=on hue=Orientation sat=Coherency bri=Original-Image ");
	// tensor: value of the standard deviation of the Gaussian local window of the structure tensor
	// gradient: index of the used gradient (0: Cubic Spline, 1: Finite difference, 2: Fourier; 3: Riesz, 4: Gaussian)
	// orientation: display the orientation map if it is set to on
	//...

	//save all images as Text Images, then close
	selectWindow("Coherency-1");
	saveAs("Text Image", fpath + " " + "coherence.txt");
	close();
	selectWindow("Orientation-1");
	saveAs("Text Image", fpath + " " + "orientation.txt");
	close();
	selectWindow("Energy-1");
	saveAs("Text Image", fpath + " " + "energy.txt");
	close();
	selectWindow(fname);
	close();
	//close OrientationJ window
	//selectWindow("OrientationJ Analysis"); 
	//run("Close"); 

	// leave a file to note that the batch worked on this directory

	File.saveString(".txt files were created by aniso_macro.ijm", parent + SENTINEL_FILE)  
}

perform_orientation_analysis(path);

//test...
//perform_orientation_analysis("C:\\Users\\David\\Desktop\\Fiji.app\\test_images\\test.tif");
