clear;
close all;

% Nombre d'images utilisees
nb_images = 36; 

% Chargement des images
im = zeros(0, 0, 0, nb_images); % Initialisation dynamique
for i = 1:nb_images
    if i <= 10
        nom = sprintf('images/viff.00%d.ppm', i-1);
    else
        nom = sprintf('images/viff.0%d.ppm', i-1);
    end
    temp_img = imread(nom);
    if i == 1
        [rows, cols, channels] = size(temp_img);
        im = zeros(rows, cols, channels, nb_images, 'uint8');
    end
    im(:,:,:,i) = temp_img;
end

% Affichage des images
figure;
subplot(2,2,1); imshow(im(:,:,:,1)); title('Image 1');
subplot(2,2,2); imshow(im(:,:,:,9)); title('Image 9');
subplot(2,2,3); imshow(im(:,:,:,17)); title('Image 17');
subplot(2,2,4); imshow(im(:,:,:,25)); title('Image 25');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choisir l'image à traiter
image_index = 1;
img = im2double(im(:,:,:,image_index));
[rows, cols, ~] = size(img);

% Conversion en espace Lab
lab_img = rgb2lab(img);

%% Segmentation en superpixels avec SLIC
K = 100; % Nombre de superpixels
m = 10; % Paramètre de compacité
step = floor(sqrt(rows * cols / K));
max_iter = 10;

% Initialisation des centres
[X, Y] = meshgrid(linspace(1, cols, round(sqrt(K))), linspace(1, rows, round(sqrt(K))));
L = interp2(1:cols, 1:rows, lab_img(:,:,1), X, Y, 'linear');
a = interp2(1:cols, 1:rows, lab_img(:,:,2), X, Y, 'linear');
b = interp2(1:cols, 1:rows, lab_img(:,:,3), X, Y, 'linear');
centers = [L(:), a(:), b(:), X(:), Y(:)];

% K-means itératif pour SLIC
for iter = 1:max_iter
    distances = inf(rows, cols);
    labels = zeros(rows, cols);

    for j = 1:size(centers, 1)
        cx = round(centers(j,4));
        cy = round(centers(j,5));

        x_min = max(cx - step, 1);
        x_max = min(cx + step, cols);
        y_min = max(cy - step, 1);
        y_max = min(cy + step, rows);

        region_L = lab_img(y_min:y_max, x_min:x_max, 1);
        region_a = lab_img(y_min:y_max, x_min:x_max, 2);
        region_b = lab_img(y_min:y_max, x_min:x_max, 3);
        [xx, yy] = meshgrid(x_min:x_max, y_min:y_max);

        color_dist = sqrt((region_L - centers(j,1)).^2 + (region_a - centers(j,2)).^2 + (region_b - centers(j,3)).^2);
        spatial_dist = sqrt((xx - cx).^2 + (yy - cy).^2);
        dist = color_dist + (m / step) * spatial_dist;

        mask = dist < distances(y_min:y_max, x_min:x_max);
        distances(y_min:y_max, x_min:x_max) = min(distances(y_min:y_max, x_min:x_max), dist);
        labels(y_min:y_max, x_min:x_max) = j .* mask;
    end

    % Mise à jour des centres
    for j = 1:size(centers, 1)
        [r, c] = find(labels == j);
        if ~isempty(r)
            centers(j, :) = [mean(lab_img(sub2ind(size(lab_img(:,:,1)), r, c))), ...
                             mean(lab_img(sub2ind(size(lab_img(:,:,2)), r, c))), ...
                             mean(lab_img(sub2ind(size(lab_img(:,:,3)), r, c))), ...
                             mean(c), mean(r)];
        end
    end

   % Affichage de l'évolution des centres des superpixels
    figure;
    imshow(img);
    hold on;
    plot(centers(:,4), centers(:,5), 'r+', 'MarkerSize', 5, 'LineWidth', 1);
    hold off;
    title(sprintf('Évolution des centres des superpixels - itération %d', iter));
    pause(0.5);
end



%% Binarisation de l'image à partir des superpixels
threshold = mean(centers(:,1)); % Seuillage basé sur la moyenne de la luminance
binary_mask = ismember(labels, find(centers(:,1) > threshold));

% Affichage du masque binaire
figure;
imshow(binary_mask);
title('Image Binarisée à partir des Superpixels');
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A FAIRE SI VOUS UTILISEZ LES MASQUES BINAIRES FOURNIS   %
% Chargement des masques binaires                         %
% de taille nb_lignes x nb_colonnes x nb_images           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ... 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A DECOMMENTER ET COMPLETER                              %
% quand vous aurez les images segmentées                  %
% Affichage des masques associes                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% subplot(2,2,1); ... ; title('Masque image 1');
% subplot(2,2,2); ... ; title('Masque image 9');
% subplot(2,2,3); ... ; title('Masque image 17');
% subplot(2,2,4); ... ; title('Masque image 25');

% chargement des points 2D suivis 
% pts de taille nb_points x (2 x nb_images)
% sur chaque ligne de pts 
% tous les appariements possibles pour un point 3D donne
% on affiche les coordonnees (xi,yi) de Pi dans les colonnes 2i-1 et 2i
% % tout le reste vaut -1
% pts = load('viff.xy');
% % Chargement des matrices de projection
% % Chaque P{i} contient la matrice de projection associee a l'image i 
% % RAPPEL : P{i} est de taille 3 x 4
% load dino_Ps;
% 
% % Reconstruction des points 3D
% X = []; % Contient les coordonnees des points en 3D
% color = []; % Contient la couleur associee
% % Pour chaque couple de points apparies
% for i = 1:size(pts,1)
%     % Recuperation des ensembles de points apparies
%     l = find(pts(i,1:2:end)~=-1);
%     % Verification qu'il existe bien des points apparies dans cette image
%     if size(l,2) > 1 & max(l)-min(l) > 1 & max(l)-min(l) < 36
%         A = [];
%         R = 0;
%         G = 0;
%         B = 0;
%         % Pour chaque point recupere, calcul des coordonnees en 3D
%         for j = l
%             A = [A;P{j}(1,:)-pts(i,(j-1)*2+1)*P{j}(3,:);
%             P{j}(2,:)-pts(i,(j-1)*2+2)*P{j}(3,:)];
%             R = R + double(im(int16(pts(i,(j-1)*2+1)),int16(pts(i,(j-1)*2+2)),1,j));
%             G = G + double(im(int16(pts(i,(j-1)*2+1)),int16(pts(i,(j-1)*2+2)),2,j));
%             B = B + double(im(int16(pts(i,(j-1)*2+1)),int16(pts(i,(j-1)*2+2)),3,j));
%         end;
%         [U,S,V] = svd(A);
%         X = [X V(:,end)/V(end,end)];
%         color = [color [R/size(l,2);G/size(l,2);B/size(l,2)]];
%     end;
% end;
% fprintf('Calcul des points 3D termine : %d points trouves. \n',size(X,2));
% 
% %affichage du nuage de points 3D
% figure;
% hold on;
% for i = 1:size(X,2)
%     plot3(X(1,i),X(2,i),X(3,i),'.','col',color(:,i)/255);
% end;
% axis equal;
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % A COMPLETER                  %
% % Tetraedrisation de Delaunay  %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % T = ...                      
% 
% % A DECOMMENTER POUR AFFICHER LE MAILLAGE
% % fprintf('Tetraedrisation terminee : %d tetraedres trouves. \n',size(T,1));
% % Affichage de la tetraedrisation de Delaunay
% % figure;
% % tetramesh(T);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % A DECOMMENTER ET A COMPLETER %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Calcul des barycentres de chacun des tetraedres
% % poids = ... 
% % nb_barycentres = ... 
% % for i = 1:size(T,1)
%     % Calcul des barycentres differents en fonction des poids differents
%     % En commencant par le barycentre avec poids uniformes
% %     C_g(:,i,1)=[ ...
% 
% % A DECOMMENTER POUR VERIFICATION 
% % A RE-COMMENTER UNE FOIS LA VERIFICATION FAITE
% % Visualisation pour vérifier le bon calcul des barycentres
% % for i = 1:nb_images
% %    for k = 1:nb_barycentres
% %        o = P{i}*C_g(:,:,k);
% %        o = o./repmat(o(3,:),3,1);
% %        imshow(im_mask(:,:,i));
% %        hold on;
% %        plot(o(2,:),o(1,:),'rx');
% %        pause;
% %        close;
% %    end
% %end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % A DECOMMENTER ET A COMPLETER %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copie de la triangulation pour pouvoir supprimer des tetraedres
% tri=T.Triangulation;
% Retrait des tetraedres dont au moins un des barycentres 
% ne se trouvent pas dans au moins un des masques des images de travail
% Pour chaque barycentre
% for k=1:nb_barycentres
% ...

% A DECOMMENTER POUR AFFICHER LE MAILLAGE RESULTAT
% Affichage des tetraedres restants
% fprintf('Retrait des tetraedres exterieurs a la forme 3D termine : %d tetraedres restants. \n',size(Tbis,1));
% figure;
% trisurf(tri,X(1,:),X(2,:),X(3,:));

% Sauvegarde des donnees
% save donnees;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONSEIL : A METTRE DANS UN AUTRE SCRIPT %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load donnees;
% Calcul des faces du maillage à garder
% FACES = ...;
% ...

% fprintf('Calcul du maillage final termine : %d faces. \n',size(FACES,1));

% Affichage du maillage final
% figure;
% hold on
% for i = 1:size(FACES,1)
%    plot3([X(1,FACES(i,1)) X(1,FACES(i,2))],[X(2,FACES(i,1)) X(2,FACES(i,2))],[X(3,FACES(i,1)) X(3,FACES(i,2))],'r');
%    plot3([X(1,FACES(i,1)) X(1,FACES(i,3))],[X(2,FACES(i,1)) X(2,FACES(i,3))],[X(3,FACES(i,1)) X(3,FACES(i,3))],'r');
%    plot3([X(1,FACES(i,3)) X(1,FACES(i,2))],[X(2,FACES(i,3)) X(2,FACES(i,2))],[X(3,FACES(i,3)) X(3,FACES(i,2))],'r');
% end;
