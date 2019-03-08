import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.control.TextField;

import java.util.ArrayList;

import javafx.event.ActionEvent;

/**
 * Class Controller - main controller class.
 * 
 * Main class that interprets the commands from the GUI, passes the command to
 * the world so it can update, and then updates the GUI based on the updated
 * world.
 * 
 * With each update to the world, the menu items in the two
 * menus---"Inventory" and "Pick up item"---are removed and then recreated using
 * the updated item information. Instead of each item 1 having a dedicated
 * ImageView object, there are three ImageView ”slots” which are populated with
 * the images of the available pick-up items whenever (1) the location changes;
 * or (2) the user takes an action to pick up or drop an item. The number of
 * items in a location is therefore capped at three, although there is no limit
 * on the number of items that can be held in inventory.
 * 
 * @author Chris Sipola (s1667278)
 * @version 2016.11.25
 */

public class Controller {

	private World world;
	private ArrayList<ImageView> itemViews = new ArrayList<>();

	@FXML
	private ImageView imageView, itemView1, itemView2, itemView3;

	@FXML
	private Button leftButton, rightButton, forwardButton;

	@FXML
	private Menu inventoryMenu, pickupItemMenu;

	@FXML
	private TextField locationDescription;

	/**
	 * Create the world and initialize its internal map.
	 */
	public Controller() {
		createWorld();
	}

	/**
	 * Initialize some variables for the world and set first view.
	 */
	public void Initialize() {

		// Add image item slots to array.
		itemViews.add(itemView1);
		itemViews.add(itemView2);
		itemViews.add(itemView3);

		updateGui(world);
	}

	/**
	 * Create views (and how they link to one another), locations and items for
	 * a created world.
	 */
	private void createWorld() {
		world = new World();
	}

	/**
	 * Turn left.
	 * 
	 * @param event
	 *            user pushes left button.
	 */
	public void turnLeft(ActionEvent event) {
		takeAction(MoveCommand.LEFT);
	}

	/**
	 * Turn right.
	 * 
	 * @param event
	 *            user pushes right button
	 */
	public void turnRight(ActionEvent event) {
		takeAction(MoveCommand.RIGHT);
	}

	/**
	 * Move forward.
	 * 
	 * @param event
	 *            user pushes forward button
	 */
	public void moveForward(ActionEvent event) {
		takeAction(MoveCommand.FORWARD);
	}

	/**
	 * Move to a new view as instructed by move command.
	 * 
	 * @param command
	 *            move command
	 */
	private void takeAction(MoveCommand command) {
		world.updateFromMoveCommand(command);
		updateGui(world);
	}

	/**
	 * Update GUI to match the world.
	 * 
	 * @param world
	 *            the world
	 */
	private void updateGui(World world) {

		// Set view image, location label text, and button clickability
		setViewImage(world);
		setTextField(world);
		disableButtons(world);

		// Set everything item-related (menus, item image).
		updateItems(world);
	}

	private void setTextField(World world) {
		locationDescription.setText(world.getCurrentLocation().getDescription());
	}

	/**
	 * Disable buttons associated with impossible actions.
	 */
	private void disableButtons(World world) {

		// Right now this just disables forward button if a move forward is
		// impossible.
		forwardButton.setDisable(!world.canMoveForward());
	}

	/**
	 * Set the image of the location view (i.e., the main image in the
	 * background).
	 */
	private void setViewImage(World world) {
		int orientation = world.getOrientation();
		Image image = world.getCurrentLocation().getLocationImage(orientation);
		imageView.setImage(image);
	}

	/**
	 * Update items (item images and item menus).
	 */
	private void updateItems(World world) {
		setItemImages(world);
		createPickupMenuItems(world);
		createInventoryMenuItems(world);
		setUsabilityOfMenus(world);
		setUsabilityOfInventoryMenuItems(world);
	}

	/**
	 * Create pick-up menu items.
	 */
	private void createPickupMenuItems(World world) {

		// First delete all menu items, then recreate below
		removeMenuItems(pickupItemMenu);
		for (Item item : world.getPickupItems()) {

			MenuItem menuItem = new MenuItem(item.getDescription());

			// When pick up menu item is clicked, add item to inventory and
			// remove from location.
			// Syntax source: https://youtu.be/AP4e6Lxncp4
			menuItem.setOnAction(e -> {
				world.getInventory().add(item);
				world.getCurrentLocation().removeItem(item);
				updateItems(world);
			});

			// Add menu item just created to pick up item menu.
			pickupItemMenu.getItems().add(menuItem);
		}
	}

	/**
	 * Create inventory menu items.
	 */
	private void createInventoryMenuItems(World world) {

		// First delete all menu items, then recreate below
		removeMenuItems(inventoryMenu);
		for (Item item : world.getInventory()) {

			MenuItem menuItem = new MenuItem(item.getDescription());

			// When menu item is clicked, remove item from inventory and
			// add to location (assuming there is room to drop).
			// Syntax source: https://youtu.be/AP4e6Lxncp4
			menuItem.setOnAction(e -> {
				if (roomToDrop(world)) {
					world.getInventory().remove(item);
					world.getCurrentLocation().addItem(item);
					updateItems(world);
				}
			});

			// Add menu item just created to inventory menu.
			inventoryMenu.getItems().add(menuItem);
		}
	}

	/**
	 * Remove items from a menu object.
	 * 
	 * @param menu
	 *            menu from which to remove all items
	 */
	private void removeMenuItems(Menu menu) {
		// Note: this was a bit complex: Iterator didn't work for this.
		// Neither did removeAll() (which doesn't remove anything) or
		// setVisible() (which gives the correct values in menus but
		// doesn't actually delete the MenuItem objects).
		while (menu.getItems().size() > 0) {
			menu.getItems().remove(0);
		}
	}

	/**
	 * Determine whether there is room to drop an item given the number of item
	 * view slots already used at the location.
	 * 
	 * @return whether there is room to drop an item
	 */
	private boolean roomToDrop(World world) {
		return world.getPickupItems().size() < itemViews.size();
	}

	/**
	 * Set the images for the items one by one in the available item image
	 * slots.
	 */
	private void setItemImages(World world) {

		// Clear all item image slots.
		for (ImageView itemView : itemViews) {
			itemView.setImage(null);
		}

		// Put item image 1 in image slot 1, item image 2 in image slot 2, etc.
		for (int i = 0; i < world.getPickupItems().size(); i++) {
			Image image = world.getPickupItems().get(i).getImage();
			itemViews.get(i).setImage(image);
		}
	}

	/**
	 * Disable menu inventory items if there is no room to drop items.
	 */
	private void setUsabilityOfInventoryMenuItems(World world) {
		if (!roomToDrop(world)) {
			for (MenuItem itemToDisable : inventoryMenu.getItems()) {
				itemToDisable.setDisable(true);
			}
		}
	}

	/**
	 * Disable the inventory menu if the user has no items. Disable the pick up
	 * item menu if there are no items to pick up.
	 */
	private void setUsabilityOfMenus(World world) {
		inventoryMenu.setDisable(world.getInventory().size() == 0);
		pickupItemMenu.setDisable(world.getPickupItems().size() == 0);
	}
}
