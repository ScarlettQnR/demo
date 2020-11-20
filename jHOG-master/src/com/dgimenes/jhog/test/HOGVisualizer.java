/*
 * The MIT License (MIT)
 * Copyright (c) 2014 Daniel Costa Gimenes
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this 
 * software and associated documentation files (the "Software"), to deal in the Software 
 * without restriction, including without limitation the rights to use, copy, modify, 
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
 * permit persons to whom the Software is furnished to do so, subject to the following 
 * conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 */
package com.dgimenes.jhog.test;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;
import javax.sql.DataSource;

import weka.core.Instances;
import weka.core.*;
import weka.core.converters.ConverterUtils.*;


import com.dgimenes.jhog.HOGProcessor;

public class HOGVisualizer {
	private static final int WIDTH_TO_SCALE = 400;

	public static void main(String[] args) throws IOException {
		File folder = new File("jHOG-master/dataset/images/in");
		for (File image : folder.listFiles()) {
			System.out.println(image.getName());
			new HOGVisualizer().processImage(image.getPath());
		}

	}

	private void processImage(String imageFilePath) throws IOException {
		BufferedImage image = ImageIO.read(new File(imageFilePath));
		HOGProcessor hog = new HOGProcessor(image);
		hog.processImage();
		this.printHOGDescriptors(hog);
		//new HOGVisualizerView(hog, WIDTH_TO_SCALE).show();

		String fileName = imageFilePath.substring(imageFilePath.lastIndexOf("\\")+1);

		String temp = imageFilePath.substring(0, imageFilePath.lastIndexOf("\\"));
		temp = temp.replace("in","out\\");
		temp = (temp+fileName+"_originalImage.png").replace(".jpg","");

		File f1 = new File(temp);

		BufferedImage originalImage =  hog.getOriginalImage();
		ImageIO.write(originalImage, "PNG", f1);

		BufferedImage luminosityImage =  hog.getLuminosityImage();
		temp=temp.replace("_originalImage","_luminosityImage");
		ImageIO.write(luminosityImage, "PNG", new File(temp));

		BufferedImage luminosityImageHEQ =  hog.getLuminosityImageHistogramEqualized();
		temp=temp.replace("_luminosityImage","_luminosityImageHEQ");
		ImageIO.write(luminosityImageHEQ, "PNG", new File(temp));

		BufferedImage luminosityImageMinMaxEq =  hog.getLuminosityImageMinMaxEqualized();
		temp=temp.replace("_luminosityImageHEQ","_luminosityImageMinMaxEq");
		ImageIO.write(luminosityImageMinMaxEq, "PNG", new File(temp));

		BufferedImage gradientMagnitude =  hog.getGradientMagnitudeImage();
		temp=temp.replace("_luminosityImageMinMaxEq","_gradientMagnitude");
		ImageIO.write(gradientMagnitude, "PNG", new File(temp));

		BufferedImage luminosityCells =  hog.getLuminosityImageWithCells(true);
		temp=temp.replace("_gradientMagnitude","_luminosityCells");
		ImageIO.write(luminosityCells, "PNG", new File(temp));

		BufferedImage hogRepresentation =  hog.getHOGDescriptorsRepresentation();
		temp=temp.replace("_luminosityCells","_hogRepresentation");
		ImageIO.write(hogRepresentation, "PNG", new File(temp));

	}

	private void printHOGDescriptors(HOGProcessor hog) {
		List<Double> hogDescriptors = hog.getHOGDescriptors();
		System.out.println("HOG DESCRIPTORS:");
		for (Double descriptor : hogDescriptors) {
			System.out.print(descriptor + "; ");
		}
	}
}
