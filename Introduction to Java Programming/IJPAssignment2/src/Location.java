import java.util.ArrayList;
import java.util.HashMap;

import javafx.scene.image.Image;

/**
 * Class Location - a location in the world.
 *
 * Holds information regarding pick-up items, images for the location, and
 * connected locations. Each location image and each connected location is
 * associated with a direction. Like orientation in World, direction is modeled
 * as cyclical from 0 to 3. The images for the locations are sourced from the
 * "images" subdirectory and set in the constructor.
 * 
 * @author Chris Sipola (s1667278)
 * @version 2016.11.25
 */

public class Location {
	private String description;
	private HashMap<Integer, Image> directionAndImage = new HashMap<>();
	private HashMap<Integer, Location> directionAndLocation = new HashMap<>();
	private ArrayList<Item> items = new ArrayList<>();

	/**
	 * Create location and add four location views using images in directory.
	 * 
	 * @param subdir
	 *            subdirectory in images directory where images are found
	 * @param description
	 *            description of the location
	 * 
	 */
	public Location(String subdir, String description) {

		this.description = description;

		// Set four location views from subdirectory of images.
		for (int i = 0; i < 4; i++) {
			Image image = new Image(String.format("/images/%s/%d.jpg", subdir, i));
			setDirectionImage(i, image);
		}
	}

	/**
	 * Set an image for a certain viewing direction.
	 * 
	 * @param direction
	 *            direction of the view
	 * @param imagePath
	 *            image path of the view image
	 */
	public void setDirectionImage(int direction, Image image) {
		directionAndImage.put(direction, image);
	}

	/**
	 * Add link from location to adjacent location to allow movement between the
	 * two.
	 * 
	 * @param direction
	 *            direction of the adjacent location
	 * @param location
	 *            adjacent location
	 */
	public void connectLocations(int direction, Location location) {
		directionAndLocation.put(direction, location);
	}

	/**
	 * Get connected location in specified direction.
	 * 
	 * @param direction
	 *            direction of connected location
	 * @return connected location
	 */
	public Location getConnectedLocation(int direction) {
		return directionAndLocation.get(direction);
	}

	/**
	 * Get image for location in specified direction.
	 * 
	 * @param direction
	 *            direction of location view
	 * @return image for view in specified direction
	 */
	public Image getLocationImage(int direction) {
		return directionAndImage.get(direction);
	}

	/**
	 * Get description of location.
	 * 
	 * @return short description of location
	 */
	public String getDescription() {
		return description;
	}

	/**
	 * Return array of items at location.
	 * 
	 * @return items at the location
	 */
	public ArrayList<Item> getItems() {
		return items;
	}

	/**
	 * Add item to location.
	 * 
	 * @param item
	 *            item to add to location
	 */
	public void addItem(Item item) {
		items.add(item);
	}

	/**
	 * Remove item from location.
	 * 
	 * @param itemToRemove
	 *            item to remove from location
	 */
	public void removeItem(Item item) {
		items.remove(item);
	}
}
