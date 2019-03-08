import javafx.scene.image.Image;

/**
 * Class Item - an item in the world.
 * 
 * Refers to an item that can be picked up or dropped. Has a description and an
 * image.
 * 
 * @author Chris Sipola (s1667278)
 * @version 2016.11.25
 */

public class Item {
	private String description;
	private Image image;

	/**
	 * Create item. Pass in Image object, not image path.
	 * 
	 * @param description
	 *            short description of item
	 * @param image
	 *            image of item
	 */
	public Item(String description, Image image) {
		this.description = description;
		this.image = image;
	}

	/**
	 * Create item. Pass in image path, not Image object.
	 * 
	 * @param description
	 *            short description of item
	 * @param imagePath
	 *            path to image of item
	 */
	public Item(String description, String imageFile) {
		this.description = description;
		this.image = new Image(String.format("/images/_items/%s", imageFile));
	}

	/**
	 * Get description of item.
	 * 
	 * @return short description of item
	 */
	public String getDescription() {
		return description;
	}

	/**
	 * Get image of item.
	 * 
	 * @return image of item
	 */
	public Image getImage() {
		return image;
	}
}
