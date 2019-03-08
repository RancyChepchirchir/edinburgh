// IJP Assignment 1, Version 6.1.2, 12 Oct 2016
package ijp.controller;

import ijp.Picture;
import ijp.service.Service;
import ijp.service.ServiceFromProperties;
import ijp.utils.Properties;
import ijp.view.View;
import ijp.view.ViewFromProperties;

import java.util.HashMap;

/**
 * Implement a controller for the PictureViewer application (uses code from given template).
 * 
 * @author Chris Sipola
 * @version 1.0
 */
public class MyController implements Controller {

	private View view;
	private Service service;
	private HashMap<Integer, String> selectionHmap = new HashMap<Integer, String>();
	
	/**
	 * Start the controller.
	 */
	
	public void start() {

		// create the view and the service objects
		view = new ViewFromProperties(this);
		service = new ServiceFromProperties();
		
		// create selections in the interface
		String[] subjects = Properties.get("MyController.subjects").split(",");
		for (String subject : subjects) {
			addSubject(subject.trim());
		}
		
		// start the interface
		view.start();
	}

	/**
	 * Handle the specified selection from the interface.
	 *
	 * @param selectionID the id of the selected item
	 */
	public void select(int selectionID) {
		
		// a picture corresponding to the selectionID
		// by default, this is an empty picture
		// (this is used if the selectionID does not match)
		Picture picture = new Picture();

		// create a picture corresponding to the selectionID
		picture = service.getPicture(selectionHmap.get(selectionID), 1);
		
		// show the picture in the interface
		view.showPicture(picture);
	}
	
	/**
	 * Add subject to the interface and to hashmap for selection 
	 * 
	 * @param subject the name of the subject
	 * 
	 * @return subjectID the id for the subject
	 */
	public int addSubject(String subject) {
		
		int selectionID = view.addSelection(subject);
		selectionHmap.put(selectionID, subject);
		
		return selectionID;
	}
}